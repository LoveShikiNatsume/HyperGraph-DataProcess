import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import re
import utils as U
from .base_dataset import BaseDataset
from datetime import datetime


def normalize_and_diff(df, column):
    try:
        ts = df[column].tolist()
        nts = np.array(ts)
        nts = (nts - nts.mean()) / (nts.std() + 0.00001)
        nts = [0] + np.diff(nts).tolist()
        df[column] = nts
        return df
    except Exception as e:
        print(repr(e))


#多线程 生成特征向量矩阵的的过程
def meta_load_metric(
    metric_df,
    instances,
    cnts,
    kpi_list,
    st_time,
    sample_interval,
):
    try:
        kpi_dict = {kpi_name: index for index, kpi_name in enumerate(kpi_list)}
        metric = []
        for instance in instances:
            ins_ts = []
            ins_df = metric_df.query(f"cmdb_id == '{instance}'")
            kpi_features = [0] * len(kpi_list)
            lst_time = st_time
            led_time = lst_time + sample_interval
            sample_df = ins_df.query(
                f"timestamp >= {lst_time} & timestamp < {led_time}"
            )
            if len(sample_df) != 0:
                for kpi_name, kpi_group in sample_df.groupby(by="kpi_name"):
                    if kpi_dict.get(kpi_name, None) is None:
                        continue
                    else:
                        kpi_features[kpi_dict[kpi_name]] = kpi_group["value"].mean()
            ins_ts.append(kpi_features)
            metric.append(kpi_features)
    except Exception as e:
        print(repr(e))
    return metric


class UnifiedMetricDataset(BaseDataset):
    def __init__(self, config) -> None:
        super().__init__(config, "metric")
        self.sample_interval = config["sample_interval"]
        
        # 从配置中加载数据集特定参数
        self.dates = config.get("dates", [])
        self.anomaly_dict = config.get("anomaly_dict", {})
        self.file_patterns = config.get("file_patterns", {})

        # feature
        self.kpi_num = 0
        self.kpi_list = []

    def load_from_tmp(self):
        appendix = super().load_from_tmp()
        self.kpi_num = appendix["kpi_num"]
        self.kpi_list = appendix["kpi_list"]

    def save_to_tmp(self):
        super().save_to_tmp({"kpi_num": self.kpi_num, "kpi_list": self.kpi_list})

    def __get_kpi_list__(self, metric_files):
        """获取KPI指标列表"""
        metric_df = self.__load_df__(metric_files)
        self.kpi_list = list(set(metric_df["kpi_name"].tolist()))
        self.kpi_list.sort()
        self.kpi_num = len(self.kpi_list)

    def __load_labels__(self, gt_df: pd.DataFrame):
        """加载标签数据"""
        r"""
        gt_df:
            columns = ["failure_type", "root_cause(service level)"]
        """
        failure_type = []
        root_cause = []
        for _, case in gt_df.iterrows():
            root_cause.append(self.services.index(case["root_cause"]))
            failure_type.append(self.failures.index(case["failure_type"]))
        self.__y__["failure_type"].extend(failure_type)
        self.__y__["root_cause"].extend(root_cause)

    def __load_metric__(self, gt_df: pd.DataFrame, metric_df: pd.DataFrame):
        """加载指标数据"""
        r"""
        gt_df:
            columns = ["st_time(10)", "ed_time(10)", "failure_type", "root_cause"]
        metric_df:
            columns = ["timestamp(10)", "cmdb_id", "kpi_name", "value"]
        """
        U.notice("Load metric")
        metric_df = metric_df.set_index("timestamp")
        pool = Pool(min(self.num_workers, len(gt_df)))
        scheduler = tqdm(total=len(gt_df), desc="dispatch")
        tasks = []
        for index, case in gt_df.iterrows():
            st_time = case["st_time"]
            ed_time = case["ed_time"]
            cnts = int((ed_time - st_time) / self.sample_interval)
            tmp_metric_df = metric_df.query(
                f"timestamp >= {st_time} & timestamp < {ed_time}"
            )
            """
            process metric
            """
            task = pool.apply_async(
                meta_load_metric,
                (
                    tmp_metric_df,#当前时间段的metric_df子集
                    self.instances,#所有服务实例
                    cnts,#时间段数
                    self.kpi_list,#kpi数量，特征向量维度
                    st_time,#gt的开始时间
                    self.sample_interval,
                ),
            )
            tasks.append(task)
            scheduler.update(1)
        pool.close()
        scheduler.close()
        scheduler = tqdm(total=len(tasks), desc="aggregate")
        for task in tasks:
            self.__X__.append(task.get())
            scheduler.update(1)
        scheduler.close()

    def get_feature(self):
        return self.kpi_num

    def rebuild(self, metric_df):
        """重建时序数据"""
        U.notice("Rebuild ts data")
        pool = Pool(self.num_workers)
        scheduler = tqdm(total=len(metric_df), desc="dispatch")
        tasks = []
        for _, cmdb_group in metric_df.groupby("cmdb_id"):
            for _, kpi_group in cmdb_group.groupby("kpi_name"):
                task = pool.apply_async(normalize_and_diff, (kpi_group, "value"))
                tasks.append(task)
                scheduler.update(len(kpi_group))
        pool.close()
        scheduler.close()
        scheduler = tqdm(total=len(metric_df), desc="aggregate")
        dfs = []
        for task in tasks:
            df = task.get()
            dfs.append(df)
            scheduler.update(len(df))
        scheduler.close()
        return pd.concat(dfs)

    def __load_groundtruth_df__(self, file_list):
        """处理groundtruth数据，统一输出格式"""
        if not file_list:
            return None
        
        # 读取文件，是否为JSON格式根据数据集决定
        groundtruth_df = self.__load_df__(file_list, is_json=self.dataset == "aiops22")
        
        # 标准化列名
        if "anomaly_type" in groundtruth_df.columns:
            groundtruth_df = groundtruth_df.rename(columns={"anomaly_type": "failure_type"})
        if "instance" in groundtruth_df.columns:
            groundtruth_df = groundtruth_df.rename(columns={"instance": "root_cause"})
        if "cmdb_id" in groundtruth_df.columns and "root_cause" not in groundtruth_df.columns:
            groundtruth_df = groundtruth_df.rename(columns={"cmdb_id": "root_cause"})
                
        # 处理时间字段
        if "st_time" in groundtruth_df.columns and isinstance(groundtruth_df["st_time"].iloc[0], str):
            groundtruth_df["st_time"] = groundtruth_df["st_time"].apply(
                lambda x: datetime.strptime(x.split(".")[0], "%Y-%m-%d %H:%M:%S").timestamp()
            )
        
        # 设置时间窗口
        duration = 600
        groundtruth_df["ed_time"] = groundtruth_df["st_time"] + duration
        
        # 故障类型映射
        if "failure_type" in groundtruth_df.columns and self.anomaly_dict:
            groundtruth_df.loc[:, "failure_type"] = groundtruth_df["failure_type"].apply(
                lambda x: self.anomaly_dict.get(x, x)
            )
        
        return groundtruth_df.loc[:, ["st_time", "ed_time", "failure_type", "root_cause"]]

    def __load_metric_df__(self, file_list):
        """处理指标数据，统一输出格式"""
        if not file_list:
            return None
        
        metric_df = self.__load_df__(file_list)
        
        # 对时间戳进行标准化处理，确保为10位整数
        if "timestamp" in metric_df.columns:
            metric_df["timestamp"] = metric_df["timestamp"].apply(
                lambda x: int(x/1000) if len(str(int(x))) > 10 else int(x)
            )
        
        # 重建时序数据
        metric_df = self.rebuild(metric_df)
        return metric_df.loc[:, ["timestamp", "cmdb_id", "kpi_name", "value"]]

    def load(self):
        """统一加载不同数据集的指标数据"""
        # 获取文件匹配模式
        metric_pattern = self.file_patterns.get("metric", ["cpu", "memory", "rx", "tx"])
        groundtruth_pattern = self.file_patterns.get("groundtruth", "groundtruth")
        
        # 获取文件列表
        groundtruth_files = self.__get_files__(self.dataset_dir, groundtruth_pattern)
        metric_files = self.__get_files__(self.dataset_dir, metric_pattern)
        
        # 获取KPI列表
        self.__get_kpi_list__(metric_files)
        
        # 按日期分组
        groundtruths = self.__add_by_date__(groundtruth_files, self.dates)
        metrics = self.__add_by_date__(metric_files, self.dates)
        
        # 逐日期处理数据
        for index, date in enumerate(self.dates):
            U.notice(f"Loading... {date}")
            
            # 检查groundtruth文件是否存在
            if not groundtruths[index]:
                U.notice(f"Skipping {date}: groundtruth_df is empty.")
                continue
            
            # 加载groundtruth
            gt_df = self.__load_groundtruth_df__(groundtruths[index])
            if gt_df is None or gt_df.empty:
                continue
            
            # 检查指标文件是否存在
            if not metrics[index]:
                U.notice(f"Skipping {date}: metric_df is empty.")
                continue
            
            # 加载指标
            metric_df = self.__load_metric_df__(metrics[index])
            
            # 处理数据
            self.__load_labels__(gt_df)
            self.__load_metric__(gt_df, metric_df)

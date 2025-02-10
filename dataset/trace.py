from tqdm import tqdm
import pandas as pd
import numpy as np
from datetime import datetime
from .base_dataset import BaseDataset
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TraceDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config, "trace")
        self.time_window = 60  # 时间窗口大小(秒)
        self.sequence_length = 10  # 序列长度，即每个特征向量包含多少个时间窗口

    def z_score_normalize(self, X):
        """
        对特征向量进行Z-score标准化
        Args:
            X: numpy array of shape (n_samples, sequence_length)
        Returns:
            标准化后的特征向量
        """
        X = np.array(X)
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        normalized_X = (X - mean) / (std + 1e-10)  # 添加小量避免除零
        return normalized_X

    def __create_time_series__(self, trace_df, start_time, end_time):
        """
        将trace数据转换为基于时间窗口的延迟序列

        Args:
            trace_df: 原始trace数据
            start_time: 开始时间戳
            end_time: 结束时间戳

        Returns:
            时间序列特征向量
        """
        # 创建时间窗口区间
        time_windows = np.arange(start_time, end_time, self.time_window)

        # 初始化序列数据
        duration_series = []

        # 对每个时间窗口计算平均延迟
        for window_start in time_windows:
            window_end = window_start + self.time_window
            # 获取窗口内的数据
            window_df = trace_df.query(f"timestamp >= {window_start} & timestamp < {window_end}")

            if len(window_df) > 0:
                # 计算平均延迟
                avg_duration = (window_df["ed_time"] - window_df["st_time"]).mean()
                duration_series.append(avg_duration)
            else:
                # 如果窗口内没有数据，填充0
                duration_series.append(0)

        # 如果序列长度不足，补0
        if len(duration_series) < self.sequence_length:
            duration_series.extend([0] * (self.sequence_length - len(duration_series)))
        # 如果序列太长，截取最后sequence_length个点
        elif len(duration_series) > self.sequence_length:
            duration_series = duration_series[-self.sequence_length:]

        return duration_series

    def __load__(self, traces, groundtruths):
        total = 0

        # 获取所有实例列表
        trace_lists = [t[0] for t in traces if t]
        if trace_lists:
            all_trace_df = self.__load_trace_df__(trace_lists)
            self.instance_list = list(set(all_trace_df["service_name"].tolist()))
        else:
            self.instance_list = []

        # 零序列用于没有数据的情况
        zero_series = [0] * self.sequence_length

        valid_data = []  # 存储有trace数据的特征向量
        final_data = []  # 存储最终的所有特征向量

        for index, date in enumerate(self.dates):
            # 加载groundtruth数据
            if groundtruths[index]:
                gt_df = self.__load_groundtruth_df__(groundtruths[index])
                total += len(gt_df)
                # print(total)

                # 如果有trace数据，处理时间序列
                if traces[index]:
                    trace_df = self.__load_trace_df__(traces[index])

                    for _, row in tqdm(
                            gt_df.iterrows(), total=len(gt_df),
                            desc=f"{date} Trace Time Series Extracting"
                    ):
                        start = row["st_time"]
                        end = row["ed_time"]

                        # 提取时间序列特征
                        time_series = self.__create_time_series__(
                            trace_df, start, end
                        )

                        valid_data.append(time_series)
                        final_data.append(time_series)
                        self.__y__["failure_type"].append(
                            int(self.failures.index(row["failure_type"]))
                        )
                        self.__y__["root_cause"].append(
                            int(self.instances.index(row["root_cause"]))
                        )

                # 没有trace数据时使用零序列
                else:
                    print(f"Warning! No trace data for date {date}")
                    for _ in gt_df.iterrows():
                        final_data.append(zero_series)
                        self.__y__["failure_type"].append(
                            int(self.failures.index(row["failure_type"]))
                        )
                        self.__y__["root_cause"].append(
                            int(self.instances.index(row["root_cause"]))
                        )

        # 只对有效数据进行归一化
        if valid_data:
            normalized_valid = self.z_score_normalize(valid_data)

            # 将归一化后的数据放回对应位置
            valid_idx = 0
            for i in range(len(final_data)):
                if not all(x == 0 for x in final_data[i]):  # 如果不是零向量
                    final_data[i] = normalized_valid[valid_idx].tolist()
                    valid_idx += 1

        self.__X__ = final_data

    def get_feature(self):
        return self.sequence_length if self.__X__ else 0

class Aiops22Trace(TraceDataset):
    def __init__(self, config):
        super().__init__(config)
        self.dates = [
            "2022-05-01",
            "2022-05-03",
            "2022-05-05",
            "2022-05-07",
            "2022-05-09",
        ]
        self.ANOMALY_DICT = {
            "k8s容器网络延迟": "network",
            "k8s容器写io负载": "io",
            "k8s容器读io负载": "io",
            "k8s容器cpu负载": "cpu",
            "k8s容器网络资源包重复发送": "network",
            "k8s容器进程中止": "process",
            "k8s容器网络丢包": "network",
            "k8s容器内存负载": "memory",
            "k8s容器网络资源包损坏": "network",
        }

    def __load_groundtruth_df__(self, file_list):
        groundtruth_df = self.__load_df__(file_list, is_json=True)
        groundtruth_df = groundtruth_df.query("level != 'node'")
        groundtruth_df.loc[:, "cmdb_id"] = groundtruth_df["cmdb_id"].apply(
            lambda x: re.sub(r"\d?-\d", "", x)
        )
        groundtruth_df = groundtruth_df.rename(columns={"timestamp": "st_time"})
        duration = 600
        groundtruth_df["ed_time"] = groundtruth_df["st_time"] + duration
        groundtruth_df["st_time"] = groundtruth_df["st_time"] - duration
        groundtruth_df = groundtruth_df.reset_index(drop=True)
        groundtruth_df.loc[:, "failure_type"] = groundtruth_df["failure_type"].apply(
            lambda x: self.ANOMALY_DICT[x]
        )
        return groundtruth_df

    def __load_trace_df__(self, file_list):
        # 读取 data
        trace_df = self.__load_df__(file_list)

        # 处理 span
        trace_df["timestamp"] = trace_df["timestamp"].apply(lambda x: int(x / 1000))
        trace_df = trace_df.rename(columns={"parent_span": "parent_id"})

        # 父子拼接
        meta_df = trace_df[["parent_id", "cmdb_id"]].rename(
            columns={"parent_id": "span_id", "cmdb_id": "ccmdb_id"}
        )
        trace_df = pd.merge(trace_df, meta_df, on="span_id")

        # 划分 span
        trace_df = trace_df.set_index("timestamp")
        trace_df = trace_df.sort_index()
        trace_df["invoke_link"] = trace_df["cmdb_id"] + "_" + trace_df["ccmdb_id"]
        return trace_df

    def load(self):
        trace_files = self.__get_files__(self.dataset_dir, "trace_jaeger-span")
        traces = self.__add_by_date__(trace_files, self.dates)
        groundtruth_files = self.__get_files__(self.dataset_dir, "groundtruth-")
        groundtruths = self.__add_by_date__(groundtruth_files, self.dates)

        self.__load__(traces, groundtruths)


class PlatformTrace(TraceDataset):
    def __init__(self, config):
        super().__init__(config)
        self.dates = [
            # "2024-03-21",
            "2024-03-22",
            "2024-03-23",
            "2024-03-24",
        ]
        self.ANOMALY_DICT = {
            "cpu anomaly": "cpu",
            "http/grpc request abscence": "http/grpc",
            "http/grpc requestdelay": "http/grpc",
            "memory overload": "memory",
            "network delay": "network",
            "network loss": "network",
            "pod anomaly": "pod_failure",
        }

    def __load_groundtruth_df__(self, file_list):
        groundtruth_df = self.__load_df__(file_list).rename(
            columns={
                "故障类型": "failure_type",
                "对应服务": "cmdb_id",
                "起始时间戳": "st_time",
                "截止时间戳": "ed_time",
                "持续时间": "duration",
            }
        )

        def meta_transfer(item):
            if item.find("(") != -1:
                item = eval(item)
                item = item[0]
            return item

        groundtruth_df.loc[:, "cmdb_id"] = groundtruth_df["cmdb_id"].apply(
            meta_transfer
        )
        groundtruth_df = groundtruth_df.rename(columns={"cmdb_id": "root_cause"})
        duration = 600
        groundtruth_df["ed_time"] = groundtruth_df["st_time"] + duration
        groundtruth_df["st_time"] = groundtruth_df["st_time"] - duration
        groundtruth_df = groundtruth_df.reset_index(drop=True)
        groundtruth_df.loc[:, "failure_type"] = groundtruth_df["failure_type"].apply(
            lambda x: self.ANOMALY_DICT[x]
        )
        return groundtruth_df

    def __load_trace_df__(self, file_list):
        # 读取 data
        trace_df = self.__load_df__(file_list)

        # 处理 span
        trace_df["timestamp"] = trace_df["timestamp"].apply(lambda x: int(x / 1e6))
        trace_df = trace_df.rename(columns={"parent_span": "parent_id"})

        # 父子拼接
        meta_df = trace_df[["parent_id", "cmdb_id"]].rename(
            columns={"parent_id": "span_id", "cmdb_id": "ccmdb_id"}
        )
        trace_df = pd.merge(trace_df, meta_df, on="span_id")

        # 划分 span
        trace_df = trace_df.set_index("timestamp")
        trace_df = trace_df.sort_index()
        trace_df["invoke_link"] = trace_df["cmdb_id"] + "_" + trace_df["ccmdb_id"]
        return trace_df

    def load(self):
        trace_files = self.__get_files__(self.dataset_dir, "trace")
        traces = self.__add_by_date__(trace_files, self.dates)
        groundtruth_files = self.__get_files__(self.dataset_dir, "ground_truth")
        groundtruths = self.__add_by_date__(groundtruth_files, self.dates)

        self.__load__(traces, groundtruths)


class GaiaTrace(TraceDataset):
    def __init__(self, config):
        super().__init__(config)
        self.dates = ["2021-07-04", "2021-07-05", "2021-07-06", "2021-07-07", "2021-07-08", "2021-07-09", "2021-07-10", "2021-07-11", "2021-07-12", "2021-07-13", "2021-07-14", "2021-07-15", "2021-07-16", "2021-07-17", "2021-07-18",  "2021-07-20", "2021-07-21", "2021-07-22", "2021-07-23", "2021-07-24", "2021-07-25", "2021-07-26", "2021-07-27", "2021-07-28", "2021-07-29", "2021-07-30", "2021-07-31"]

        self.ANOMALY_DICT = {
            "[memory_anomalies]": "memory",
            "[normal memory freed label]": "memory",
            "[access permission denied exception]": "access",
            "[login failure]": "login",
            "[file moving program]": "file",
        }

    def __load_groundtruth_df__(self, file_list):
        groundtruth_df = self.__load_df__(file_list).rename(
            columns={
                "anomaly_type": "failure_type",
                "instance": "instance_name",
            }
        )

        def meta_transfer(item):
            if item.find("(") != -1:
                item = eval(item)
                item = item[0]
            return item

        groundtruth_df.loc[:, "instance_name"] = groundtruth_df["instance_name"].apply(meta_transfer)
        groundtruth_df = groundtruth_df.rename(columns={"instance_name": "root_cause"})
        duration = 600
        groundtruth_df["st_time"] = groundtruth_df["st_time"].apply(
            lambda x: datetime.strptime(
                x.split(".")[0], "%Y-%m-%d %H:%M:%S"
            ).timestamp()
        )
        groundtruth_df["ed_time"] = groundtruth_df["ed_time"].apply(
            lambda x: datetime.strptime(
                x.split(".")[0], "%Y-%m-%d %H:%M:%S"
            ).timestamp()
        )
        groundtruth_df["ed_time"] = groundtruth_df["st_time"] + duration
        groundtruth_df = groundtruth_df.reset_index(drop=True)
        groundtruth_df.loc[:, "failure_type"] = groundtruth_df["failure_type"].apply(
            lambda x: self.ANOMALY_DICT[x]
        )
        return groundtruth_df

    def __load_trace_df__(self, file_list):
        trace_df = self.__load_df__(file_list)

        trace_df = trace_df.set_index("timestamp")
        trace_df = trace_df.sort_index()

        trace_df["duration"] = trace_df["ed_time"] - trace_df["st_time"]
        return trace_df

    def load(self):
        trace_files = self.__get_files__(self.dataset_dir, "trace")
        traces = self.__add_by_date__(trace_files, self.dates)

        groundtruth_files = self.__get_files__(self.dataset_dir, "groundtruth")
        groundtruths = self.__add_by_date__(groundtruth_files, self.dates)

        self.__load__(traces, groundtruths)
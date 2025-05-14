from tqdm import tqdm
import pandas as pd
import numpy as np
from datetime import datetime
from .base_dataset import BaseDataset
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class UnifiedTraceDataset(BaseDataset):
    def __init__(self, config) -> None:
        super().__init__(config, "trace")
        self.time_window = 60  # 时间窗口大小(秒)
        self.sequence_length = 10  # 序列长度，即每个特征向量包含多少个时间窗口

        # 从配置中加载数据集特定参数
        self.dates = config.get("dates", [])
        self.anomaly_dict = config.get("anomaly_dict", {})
        self.file_patterns = config.get("file_patterns", {})

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
            
        # 如果有ed_time列且为字符串格式，转换为时间戳
        if "ed_time" in groundtruth_df.columns and isinstance(groundtruth_df["ed_time"].iloc[0], str):
            groundtruth_df["ed_time"] = groundtruth_df["ed_time"].apply(
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
        
        return groundtruth_df

    def __load_trace_df__(self, file_list):
        """处理调用链数据，统一输出格式"""
        if not file_list:
            return None
        
        trace_df = self.__load_df__(file_list)
        
        # 标准化时间戳
        if "timestamp" in trace_df.columns:
            # 确保为10位整数时间戳
            trace_df["timestamp"] = trace_df["timestamp"].apply(
                lambda x: int(x/1000) if len(str(int(x))) > 10 else int(x)
            )
        
        # 处理特定字段
        if "parent_span" in trace_df.columns:
            trace_df = trace_df.rename(columns={"parent_span": "parent_id"})
            
            # 父子拼接
            if "parent_id" in trace_df.columns and "span_id" in trace_df.columns:
                meta_df = trace_df[["parent_id", "cmdb_id"]].rename(
                    columns={"parent_id": "span_id", "cmdb_id": "ccmdb_id"}
                )
                trace_df = pd.merge(trace_df, meta_df, on="span_id")
                
                # 添加调用链关系
                if "cmdb_id" in trace_df.columns and "ccmdb_id" in trace_df.columns:
                    trace_df["invoke_link"] = trace_df["cmdb_id"] + "_" + trace_df["ccmdb_id"]
        
        # 设置索引和排序
        trace_df = trace_df.set_index("timestamp")
        trace_df = trace_df.sort_index()
        
        # 计算持续时间
        if "st_time" in trace_df.columns and "ed_time" in trace_df.columns:
            trace_df["duration"] = trace_df["ed_time"] - trace_df["st_time"]
        
        return trace_df

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

    def load(self):
        """统一加载不同数据集的调用链数据"""
        # 获取文件匹配模式
        trace_pattern = self.file_patterns.get("trace", "trace")
        groundtruth_pattern = self.file_patterns.get("groundtruth", "groundtruth")

        # 获取文件列表
        trace_files = self.__get_files__(self.dataset_dir, trace_pattern)
        groundtruth_files = self.__get_files__(self.dataset_dir, groundtruth_pattern)

        # 按日期分组
        traces = self.__add_by_date__(trace_files, self.dates)
        groundtruths = self.__add_by_date__(groundtruth_files, self.dates)

        # 处理数据
        self.__load__(traces, groundtruths)
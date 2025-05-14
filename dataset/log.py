import json
from transformers import BertTokenizer, BertModel
import torch
from .drain3.file_persistence import FilePersistence
from .drain3.template_miner import TemplateMiner
from .drain3.template_miner_config import TemplateMinerConfig
import pandas as pd
from tqdm import tqdm
from typing import *
import utils as U
from .base_dataset import BaseDataset
import re
import numpy as np
from typing import Any
import random
from datetime import datetime

class BertEncoder:
    def __init__(self, config) -> None:
        self._bert_tokenizer = BertTokenizer.from_pretrained(config["tokenizer_path"])
        self._bert_model = BertModel.from_pretrained(config["model_path"])
        self.cache = {}

    def __call__(self, sentence, no_wordpiece=False) -> Any:
        r"""
        return list(len=768)
        """
        if self.cache.get(sentence, None) is None:
            if no_wordpiece:
                words = sentence.split(" ")
                words = [
                    word for word in words if word in self._bert_tokenizer.vocab.keys()
                ]
                sentence = " ".join(words)
            inputs = self._bert_tokenizer(
                sentence, truncation=True, return_tensors="pt", max_length=512
            )
            outputs = self._bert_model(**inputs)

            embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze(dim=1)
            self.cache[sentence] = embedding[0].tolist()
            return embedding[0].tolist()
        else:
            return self.cache[sentence]

class DrainProcesser:
    def __init__(self, config) -> None:
        r"""
        config: {
            "save_path": "path/to",
            "drain_config_path": "path/to"
        }
        """
        self._drain_config_path = config["drain_config_path"]
        U.check(config["drain_save_path"])
        persistence = FilePersistence(config["drain_save_path"])
        miner_config = TemplateMinerConfig()
        miner_config.load(config["drain_config_path"])
        self._template_miner = TemplateMiner(persistence, config=miner_config)

    def __call__(self, sentence) -> str:
        line = str(sentence).strip()
        result = self._template_miner.add_log_message(line)
        return result["template_mined"]

class UnifiedLogDataset(BaseDataset):
    def __init__(self, config) -> None:
        super().__init__(config, "log")
        self.sample_interval = self.__config__["sample_interval"]
        self._drain = DrainProcesser(config["drain_config"])
        self._encoder = BertEncoder(config["bert_config"])
        
        # 从配置中加载数据集特定参数
        self.dates = config.get("dates", [])
        self.anomaly_dict = config.get("anomaly_dict", {})
        self.file_patterns = config.get("file_patterns", {})

    #处理一个故障注入记录，分为多个时间窗口，一个60秒，cnts表示时间窗口数量，用于评估日志模板权重
    def __add_sample__(self, st_time, cnts, log_df, label):
        #该方法中调用了drain以及Bert，编码为特征向量；相当于metric的meta_load_metric
        cnt_of_log = {}
        seqs = []
        for cnt in range(cnts):
            lst_time = st_time + self.sample_interval * cnt
            led_time = lst_time + self.sample_interval
            sample_df = log_df.query(
                f"timestamp >= {lst_time} & timestamp < {led_time}"
            )
            template_list = []
            for log in sample_df["message"].tolist():
                template = self._drain(log)
                if cnt_of_log.get(template, None) is None:
                    cnt_of_log[template] = [0] * cnts
                #每个模板在每个时间段内出现的频率
                cnt_of_log[template][cnt] += 1
                template_list.append(template)
                #该时间段内所有不同模板的集合
            seqs.append(list(set(template_list)))
        # with open(f"log_tmp/{st_time}.json", "w", encoding="utf8") as w:
        #     json.dump(cnt_of_log, w)
        wei_of_log = {}
        total_gap = 0.00001
        for template, cnt_list in cnt_of_log.items():
            cnt_list = np.array(cnt_list)
            cnt_list = np.log(cnt_list + 0.00001)
            cnt_list = np.abs([0] + np.diff(cnt_list))
            gap = cnt_list.max() - cnt_list.mean()
            wei_of_log[template] = gap
            total_gap += gap

        final_repr = np.zeros((768,))
        for seq in seqs:
            window_repr = np.zeros((768,))
            for template in seq:
                window_repr += (
                    wei_of_log[template] * np.array(self._encoder(template)) / total_gap
                )
            final_repr += window_repr

        final_repr = final_repr / cnts
        self.__X__.append(final_repr.tolist())
        self.__y__["failure_type"].append(label["failure_type"])
        self.__y__["root_cause"].append(label["root_cause"])

    def __add_zero_sample__(self, st_time, cnts, label):
        """为没有日志数据的groundtruth添加全零特征向量"""
        final_repr = np.zeros((768,))
        self.__X__.append(final_repr.tolist())
        self.__y__["failure_type"].append(label["failure_type"])
        self.__y__["root_cause"].append(label["root_cause"])

    def __load__(self, log_df: pd.DataFrame, groundtruth_df: pd.DataFrame):
        r"""
        :log_df       : [timestamp, message]
        :groundtruth_df   : [st_time, ed_time, failure_type, root_cause]
        """
        log_columns = log_df.columns.tolist()
        assert "timestamp" in log_columns, "log_df requires `timestamp`"
        assert "message" in log_columns, "log_df requires `message`"
        log_df = log_df.sort_values(by="timestamp")
        log_df["label"] = 0
        anomaly_columns = groundtruth_df.columns.tolist()
        assert "st_time" in anomaly_columns, "groundtruth_df requires `st_time`"
        assert "ed_time" in anomaly_columns, "groundtruth_df requires `ed_time`"
        assert (
            "failure_type" in anomaly_columns
        ), "groundtruth_df requires `failure_type`"
        assert "root_cause" in anomaly_columns, "groundtruth_df requires `root_cause`"
        st_time = log_df.head(1)["timestamp"].item()
        ed_time = log_df.tail(1)["timestamp"].item()

        process_bar = tqdm(
            total=len(groundtruth_df),
            desc=f"process {self.__desc_date__(st_time)} ~ {self.__desc_date__(ed_time)}",
        )
        for index, case in groundtruth_df.iterrows():
            in_condition = log_df.loc[
                (log_df["timestamp"] >= case["st_time"])
                & (log_df["timestamp"] <= case["ed_time"]),
                :,
            ]
            cnts = int((case["ed_time"] - case["st_time"]) / self.sample_interval)
            self.__add_sample__(
                case["st_time"],
                cnts,
                in_condition,
                {
                    "failure_type": self.failures.index(case["failure_type"]),
                    "root_cause": self.services.index(case["root_cause"]),
                },
            )
            process_bar.update(1)
        process_bar.close()

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

    def __load_log_df__(self, file_list):
        """处理日志数据，统一输出格式"""
        if not file_list:
            return None
        
        log_df = self.__load_df__(file_list)
        
        # 标准化列名
        if "value" in log_df.columns and "message" not in log_df.columns:
            log_df = log_df.rename(columns={"value": "message"})
        if "service" in log_df.columns and "cmdb_id" not in log_df.columns:
            log_df = log_df.rename(columns={"service": "cmdb_id"})
        
        # 确保timestamp是整数类型
        if "timestamp" in log_df.columns:
            log_df["timestamp"] = log_df["timestamp"].astype(int)
        
        return log_df.loc[:, ["timestamp", "message"]]

    def load(self):
        """统一加载不同数据集的日志数据"""
        # 获取文件匹配模式
        log_pattern = self.file_patterns.get("log", "log")
        groundtruth_pattern = self.file_patterns.get("groundtruth", "groundtruth")
        
        # 获取文件列表
        groundtruth_files = self.__get_files__(self.dataset_dir, groundtruth_pattern)
        log_files = self.__get_files__(self.dataset_dir, log_pattern)
        
        # 按日期分组
        groundtruths = self.__add_by_date__(groundtruth_files, self.dates)
        logs = self.__add_by_date__(log_files, self.dates)
        
        # 逐日期处理数据
        for index, date in enumerate(self.dates):
            U.notice(f"Loading... {date}")
            
            # 加载 groundtruth_df
            if not groundtruths[index]:
                U.notice(f"Skipping {date}: groundtruth_df is empty.")
                continue

            groundtruth_df = self.__load_groundtruth_df__(groundtruths[index])
            if groundtruth_df is None or groundtruth_df.empty:
                continue
                
            # 加载 log_df，如果没有日志文件，log_df 为 None
            log_df = self.__load_log_df__(logs[index]) if logs[index] else None
            
            # 对于每条groundtruth记录
            for _, case in groundtruth_df.iterrows():
                cnts = int((case["ed_time"] - case["st_time"]) / self.sample_interval)
                label = {
                    "failure_type": self.failures.index(case["failure_type"]),
                    "root_cause": self.services.index(case["root_cause"]),
                }

                if log_df is not None:
                    # 如果有日志数据，正常处理
                    in_condition = log_df.loc[
                        (log_df["timestamp"] >= case["st_time"]) & 
                        (log_df["timestamp"] <= case["ed_time"]),
                        :,
                    ]
                    self.__add_sample__(case["st_time"], cnts, in_condition, label)
                else:
                    # 如果没有日志数据，添加全零特征向量
                    self.__add_zero_sample__(case["st_time"], cnts, label)
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

class LogDataset(BaseDataset):
    def __init__(self, config) -> None:
        #Drain、Bert工具初始化
        r"""
        X: [sample_num, seq, n_model]
        """
        super().__init__(config, "log")
        self.sample_interval = self.__config__["sample_interval"]
        self._drain = DrainProcesser(config["drain_config"])
        self._encoder = BertEncoder(config["bert_config"])

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

class Aiops22Log(LogDataset):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
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
        groundtruth_df = groundtruth_df.rename(
            columns={"timestamp": "st_time", "cmdb_id": "root_cause"}
        )
        duration = 600
        groundtruth_df["ed_time"] = groundtruth_df["st_time"] + duration
        groundtruth_df["st_time"] = groundtruth_df["st_time"] - duration
        groundtruth_df = groundtruth_df.reset_index(drop=True)
        groundtruth_df.loc[:, "failure_type"] = groundtruth_df["failure_type"].apply(
            lambda x: self.ANOMALY_DICT[x]
        )
        return groundtruth_df.loc[
            :, ["st_time", "ed_time", "failure_type", "root_cause"]
        ]

    def __load_log_df__(self, file_list):
        # read log
        log_df = self.__load_df__(file_list)
        log_df = log_df.rename(columns={"value": "message"})
        return log_df.loc[:, ["timestamp", "message"]]

    def load(self):
        # read groundtruth
        groundtruth_files = self.__get_files__(self.dataset_dir, "groundtruth-")
        log_files = self.__get_files__(self.dataset_dir, "-log-service")
        dates = [
            "2022-05-01",
            "2022-05-03",
            "2022-05-05",
            "2022-05-07",
            "2022-05-09",
        ]
        groundtruths = self.__add_by_date__(groundtruth_files, dates)
        logs = self.__add_by_date__(log_files, dates)

        for index, date in enumerate(dates):
            U.notice(f"Loading... {date}")
            groundtruth_df = self.__load_groundtruth_df__(groundtruths[index])
            log_df = self.__load_log_df__(logs[index])
            self.__load__(log_df, groundtruth_df)

class PlatformLog(LogDataset):
    def __init__(self, config) -> None:
        super().__init__(config)
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
        return groundtruth_df.loc[
            :, ["st_time", "ed_time", "failure_type", "root_cause"]
        ]

    def __load_log_df__(self, file_list):
        # read log
        log_df = self.__load_df__(file_list)
        log_df.loc[:, "timestamp"] = log_df["timestamp"].apply(lambda x: int(x))
        return log_df.loc[:, ["timestamp", "message"]]

    def load(self):
        groundtruth_files = self.__get_files__(self.dataset_dir, "ground_truth")
        log_files = self.__get_files__(self.dataset_dir, "log")
        dates = [
            # "2024-03-20",
            # "2024-03-21",
            "2024-03-22",
            "2024-03-23",
            "2024-03-24",
        ]
        groundtruths = self.__add_by_date__(groundtruth_files, dates)
        logs = self.__add_by_date__(log_files, dates)

        for index, date in enumerate(dates):
            U.notice(f"Loading... {date}")
            groundtruth_df = self.__load_groundtruth_df__(groundtruths[index])
            log_df = self.__load_log_df__(logs[index])
            self.__load__(log_df, groundtruth_df)

class GaiaLog(LogDataset):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.ANOMALY_DICT = {
            "[memory_anomalies]": "memory",
            "[normal memory freed label]": "memory",
            "[access permission denied exception]": "access",
            "[login failure]": "login",
            "[file moving program]": "file",
        }

    def __load_groundtruth_df__(self, file_list):
        r"""
        输出:
        groundtruth_df: 一个标准化后的 DataFrame，包含以下字段：
            st_time: 异常起始时间戳（秒）。
            ed_time: 异常结束时间戳（秒）。
            failure_type: 异常类型，映射到统一类别。
            root_cause: 异常根因服务实例。
        """
        #读取list中的所有csv文件内容，合并为一个DataFrame，并重命名列名
        groundtruth_df = self.__load_df__(file_list).rename(
            columns={
                "anomaly_type": "failure_type",
            }
        )
        groundtruth_df = groundtruth_df.rename(columns={"instance": "root_cause"})
        duration = 600
        from datetime import datetime

        groundtruth_df["st_time"] = groundtruth_df["st_time"].apply(
            lambda x: datetime.strptime(
                x.split(".")[0], "%Y-%m-%d %H:%M:%S"
            ).timestamp()
        )
        groundtruth_df["ed_time"] = groundtruth_df["st_time"] + duration
        #不论是11s还是600s,统一时间窗口为600s
        # groundtruth_df["st_time"] = groundtruth_df["st_time"] - duration
        groundtruth_df = groundtruth_df.reset_index(drop=True)
        groundtruth_df.loc[:, "failure_type"] = groundtruth_df["failure_type"].apply(
            lambda x: self.ANOMALY_DICT[x]
        )
        return groundtruth_df.loc[
            :, ["st_time", "ed_time", "failure_type", "root_cause"]
        ]

    def __load_log_df__(self, file_list):
        from datetime import datetime

        if not file_list:  # 如果没有日志文件
            return None

        log_df = self.__load_df__(file_list)

        def meta_ts(x):
            try:
                return datetime.strptime(
                    x.split(",")[0], "%Y-%m-%d %H:%M:%S"
                ).timestamp()
            except:
                return 0

        log_df.loc[:, "timestamp"] = log_df["message"].apply(meta_ts)
        log_df["message"] = log_df["message"].apply(
            lambda x: "|".join(str(x).split("|")[1:])
        )
        log_df = log_df.rename(columns={"service": "cmdb_id"})
        return log_df.loc[:, ["timestamp", "message"]]

    def __add_zero_sample__(self, st_time, cnts, label):
        """为没有日志数据的groundtruth添加全零特征向量"""
        final_repr = np.zeros((768,))
        self.__X__.append(final_repr.tolist())
        self.__y__["failure_type"].append(label["failure_type"])
        self.__y__["root_cause"].append(label["root_cause"])

    def load(self):
        groundtruth_files = self.__get_files__(self.dataset_dir, "groundtruth")
        log_files = self.__get_files__(self.dataset_dir, "log")
        dates = ["2021-07-04", "2021-07-05", "2021-07-06", "2021-07-07", "2021-07-08", "2021-07-09", "2021-07-10",
                 "2021-07-11", "2021-07-12", "2021-07-13", "2021-07-14", "2021-07-15", "2021-07-16", "2021-07-17",
                 "2021-07-18", "2021-07-20", "2021-07-21", "2021-07-22", "2021-07-23", "2021-07-24", "2021-07-25",
                 "2021-07-26", "2021-07-27", "2021-07-28", "2021-07-29", "2021-07-30", "2021-07-31"]
        groundtruths = self.__add_by_date__(groundtruth_files, dates)
        logs = self.__add_by_date__(log_files, dates)

        for index, date in enumerate(dates):
            U.notice(f"Loading... {date}")

            # 加载 groundtruth_df
            if not groundtruths[index]:
                U.notice(f"Skipping {date}: groundtruth_df is empty.")
                continue

            groundtruth_df = self.__load_groundtruth_df__(groundtruths[index])

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
                                   (log_df["timestamp"] >= case["st_time"])
                                   & (log_df["timestamp"] <= case["ed_time"]),
                                   :,
                                   ]
                    self.__add_sample__(case["st_time"], cnts, in_condition, label)
                else:
                    # 如果没有日志数据，添加全零特征向量
                    self.__add_zero_sample__(case["st_time"], cnts, label)
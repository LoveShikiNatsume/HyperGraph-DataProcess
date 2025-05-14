from config import CONFIG_DICT
import utils as U
from datetime import datetime
import argparse
import json
import os
import torch
import numpy as np
from dataset import LOG_DATASET, METRIC_DATASET, TRACE_DATASET
from model.encoder import LogEncoder, MetricEncoder, TraceEncoder
from model.fusion import ConcatFusion

class FeatureProcessor:
    """用于预处理不同模态数据、提取特征并融合的类"""
    def __init__(self, config, time_str):
        self.config = config
        self.time_str = time_str
        self.device = U.get_device(config)
        self.instance_map = {instance: i for i, instance in enumerate(config["instances"].split(" "))}
        
    def process_service(self, service):
        """处理指定服务的日志和调用链数据"""
        print(f"\n处理服务: {service}")
        # 更新配置中的服务名和相关路径
        self.config["service_name"] = service
        
        # 处理日志和调用链数据
        self._process_log(service)
        self._process_trace(service)
    
    def _process_log(self, service):
        """处理日志数据"""
        U.notice(f"处理{service}的日志数据...")
        
        # 创建服务特定的目录路径
        service_data_dir = os.path.join(self.config.get("service_dir", self.config["dataset_dir"]), service)
        service_save_dir = os.path.join(self.config["save_dir"], "service", service)
        
        # 打印路径信息，帮助调试
        print(f"服务 {service} 的数据目录: {service_data_dir}")
        print(f"服务 {service} 的保存目录: {service_save_dir}")
        
        # 创建目录
        os.makedirs(service_save_dir, exist_ok=True)
        
        # 加载日志数据集，通过新增方法设置服务特定路径
        dataset = LOG_DATASET[self.config["dataset"]](self.config)
        dataset.set_service_paths(service_data_dir, service_save_dir)
        
        if self.config["use_tmp"] and os.path.exists(dataset.get_dataset_path()):
            print(f"使用缓存的日志数据集")
            dataset.load_from_tmp()
        else:
            print(f"加载日志数据集")
            dataset.load()
            dataset.save_to_tmp()
        
        print(f"{service}的日志数据处理完成，共{len(dataset.X)}条记录")
    
    def _process_trace(self, service):
        """处理调用链数据"""
        U.notice(f"处理{service}的调用链数据...")
        
        # 创建服务特定的目录路径
        service_data_dir = os.path.join(self.config.get("service_dir", self.config["dataset_dir"]), service)
        service_save_dir = os.path.join(self.config["save_dir"], "service", service)
        
        # 创建目录
        os.makedirs(service_save_dir, exist_ok=True)
        
        # 加载调用链数据集，通过新增方法设置服务特定路径
        dataset = TRACE_DATASET[self.config["dataset"]](self.config)
        dataset.set_service_paths(service_data_dir, service_save_dir)
        
        if self.config["use_tmp"] and os.path.exists(dataset.get_dataset_path()):
            print(f"使用缓存的调用链数据集")
            dataset.load_from_tmp()
        else:
            print(f"加载调用链数据集")
            dataset.load()
            dataset.save_to_tmp()
        
        print(f"{service}的调用链数据处理完成，共{len(dataset.X)}条记录")
    
    def process_metrics(self):
        """处理指标数据并拆分到各服务目录"""
        U.notice("处理指标数据...")
        
        # 加载指标数据集
        dataset = METRIC_DATASET[self.config["dataset"]](self.config)
        
        if self.config["use_tmp"] and os.path.exists(dataset.get_dataset_path()):
            print(f"使用缓存的指标数据集")
            dataset.load_from_tmp()
        else:
            print(f"加载指标数据集")
            dataset.load()
            dataset.save_to_tmp()
        
        print(f"指标数据处理完成，共{len(dataset.X)}条记录")
        
        # 在指标数据处理完成后，自动拆分指标数据到各服务文件夹
        self.split_metrics_by_service()
    
    def split_metrics_by_service(self):
        """将metric数据按服务拆分并保存到对应服务文件夹，同时保留原始文件"""
        U.notice("将指标数据拆分到各服务文件夹...")
        
        # 读取原始的指标特征文件
        metric_path = os.path.join(self.config["save_dir"], f"{self.config['dataset']}_metric_tmp.json")
        if not os.path.exists(metric_path):
            print(f"找不到指标数据文件: {metric_path}")
            return
        
        with open(metric_path, 'r') as f:
            metric_data = json.load(f)
        
        # 获取服务列表
        services = self.config["instances"].split(" ")
        print(f"拆分指标数据到 {len(services)} 个服务实例")
        
        # 获取每个服务实例在特征向量中的索引
        service_indices = {service: i for i, service in enumerate(services)}
        
        # 为每个服务创建文件夹并保存对应的指标数据
        for service in services:
            # 创建服务文件夹
            service_dir = os.path.join(self.config["save_dir"], "service", service)
            os.makedirs(service_dir, exist_ok=True)
            
            # 获取该服务的特征向量
            service_idx = service_indices[service]
            service_features = []
            
            # 从原始特征向量中提取当前服务的特征
            for x in metric_data['X']:
                service_features.append(x[service_idx])
            
            # 创建服务特定的指标数据
            service_metric_data = {
                "X": service_features,
                "y": metric_data['y'],  # 保留相同的标签信息
                "appendix": {
                    "service_name": service,
                    "kpi_num": metric_data['appendix']['kpi_num'],
                    "kpi_list": metric_data['appendix']['kpi_list']
                }
            }
            
            # 保存到服务文件夹
            service_metric_path = os.path.join(service_dir, f"{self.config['dataset']}_metric_tmp.json")
            with open(service_metric_path, 'w') as f:
                json.dump(service_metric_data, f)
            
            print(f"  - 已将指标数据保存到: {service}")
        
        print(f"指标数据拆分完成（原始文件已保留）")

    def merge_features(self):
        """合并所有模态的特征"""
        U.notice("合并所有模态的特征...")
        
        # 加载指标特征
        metric_path = os.path.join(self.config["save_dir"], f"{self.config['dataset']}_metric_tmp.json")
        with open(metric_path, 'r') as f:
            metric_data = json.load(f)
        
        metric_X = np.array(metric_data['X'])
        ground_truth = metric_data['y']
        n_samples = len(ground_truth['failure_type'])
        print(f"加载指标特征：{metric_X.shape}")
        
        # 初始化日志和调用链特征矩阵
        services = self.config["instances"].split(" ")
        log_features_shape = (n_samples, len(services), 768)  # 768是BERT的特征维度
        trace_features_shape = (n_samples, len(services), 10)  # 10是调用链的特征维度
        
        log_features = np.zeros(log_features_shape)
        trace_features = np.zeros(trace_features_shape)
        
        # 加载各个服务的日志和调用链特征
        for i, service in enumerate(services):
            # 加载日志特征
            service_log_path = os.path.join(self.config["save_dir"], "service", service, f"{self.config['dataset']}_log_tmp.json")
            if os.path.exists(service_log_path):
                with open(service_log_path, 'r') as f:
                    log_data = json.load(f)
                log_X = np.array(log_data['X'])
                log_features[:, i, :] = log_X
            
            # 加载调用链特征
            service_trace_path = os.path.join(self.config["save_dir"], "service", service, f"{self.config['dataset']}_trace_tmp.json")
            if os.path.exists(service_trace_path):
                with open(service_trace_path, 'r') as f:
                    trace_data = json.load(f)
                trace_X = np.array(trace_data['X'])
                trace_features[:, i, :] = trace_X
        
        print(f"日志特征形状: {log_features.shape}")
        print(f"调用链特征形状: {trace_features.shape}")
        print(f"指标特征形状: {metric_X.shape}")
        
        # 创建汇总特征文件
        merged_features = {
            'log': log_features.tolist(),
            'metric': metric_X.tolist(),
            'trace': trace_features.tolist(),
            'y': ground_truth
        }
        
        merged_path = os.path.join(self.config["save_dir"], f"{self.config['dataset']}_merged_features.json")
        with open(merged_path, 'w') as f:
            json.dump(merged_features, f)
        
        print(f"所有特征已合并到: {merged_path}")
        
        # 自动执行两种融合方式
        self.simple_concat_features(merged_features)
        self.apply_feature_extraction(merged_features)
        
        return merged_features
    
    def apply_feature_extraction(self, features=None):
        """应用模型编码器提取更好的特征"""
        if features is None:
            merged_path = os.path.join(self.config["save_dir"], f"{self.config['dataset']}_merged_features.json")
            if not os.path.exists(merged_path):
                features = self.merge_features()
            else:
                with open(merged_path, 'r') as f:
                    features = json.load(f)
        
        U.notice("使用编码器提取特征...")
        
        # 初始化编码器
        log_encoder = LogEncoder(
            self.config["max_len"], 
            self.config["d_model"], 
            self.config["nhead"], 
            self.config["d_ff"], 
            self.config["layer_num"], 
            self.config["dropout"],
            self.device
        ).to(self.device)
        
        metric_encoder = MetricEncoder(
            len(features['metric'][0][0]),  # 指标数量
            len(self.config["instances"].split(" ")),  # 实例数量
            self.config["max_len"], 
            self.config["d_model"], 
            self.config["nhead"], 
            self.config["d_ff"], 
            self.config["layer_num"], 
            self.config["dropout"],
            self.device
        ).to(self.device)
        
        trace_encoder = TraceEncoder(
            features['trace'][0][0][0],  # 调用链特征维度
            self.config["d_model"], 
            self.config["nhead"], 
            self.config["d_ff"], 
            self.config["dropout"]
        ).to(self.device)
        
        # 提取特征
        log_features = torch.tensor(features['log'], dtype=torch.float).to(self.device)
        metric_features = torch.tensor(features['metric'], dtype=torch.float).to(self.device)
        trace_features = torch.tensor(features['trace'], dtype=torch.float).to(self.device)
        
        # 使用编码器处理批量数据
        batch_size = 32
        num_samples = log_features.shape[0]
        encoded_log = []
        encoded_metric = []
        encoded_trace = []
        
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                end = min(i + batch_size, num_samples)
                
                # 处理日志特征
                log_batch = log_features[i:end].reshape(-1, log_features.shape[2])
                log_encoded = log_encoder(log_batch)
                encoded_log.append(log_encoded.cpu().numpy())
                
                # 处理指标特征
                metric_batch = metric_features[i:end]
                metric_encoded = metric_encoder(metric_batch)
                encoded_metric.append(metric_encoded.cpu().numpy())
                
                # 处理调用链特征
                trace_batch = trace_features[i:end].reshape(-1, trace_features.shape[2])
                trace_encoded = trace_encoder(trace_batch)
                encoded_trace.append(trace_encoded.cpu().numpy())
        
        encoded_log = np.vstack(encoded_log)
        encoded_metric = np.vstack(encoded_metric)
        encoded_trace = np.vstack(encoded_trace)
        
        print(f"编码后的日志特征形状: {encoded_log.shape}")
        print(f"编码后的指标特征形状: {encoded_metric.shape}")
        print(f"编码后的调用链特征形状: {encoded_trace.shape}")
        
        # 简单拼接特征
        concatenated_features = np.hstack([encoded_log, encoded_metric, encoded_trace])
        
        # 存储结果
        encoded_features = {
            'encoded_log': encoded_log.tolist(),
            'encoded_metric': encoded_metric.tolist(),
            'encoded_trace': encoded_trace.tolist(),
            'concatenated': concatenated_features.tolist(),
            'y': features['y']
        }
        
        encoded_path = os.path.join(self.config["save_dir"], f"{self.config['dataset']}_encoded_features.json")
        with open(encoded_path, 'w') as f:
            json.dump(encoded_features, f)
        
        print(f"编码后的特征已保存到: {encoded_path}")
        
        return encoded_features
    
    def simple_concat_features(self, features=None):
        """简单拼接三种模态的特征，不经过编码器"""
        if features is None:
            merged_path = os.path.join(self.config["save_dir"], f"{self.config['dataset']}_merged_features.json")
            if not os.path.exists(merged_path):
                features = self.merge_features()
            else:
                with open(merged_path, 'r') as f:
                    features = json.load(f)
        
        U.notice("简单拼接特征...")
        
        # 将各服务的数据平均或合并
        log_features = np.array(features['log'])
        metric_features = np.array(features['metric'])
        trace_features = np.array(features['trace'])
        
        # 对每个样本，拼接所有服务的特征
        n_samples = log_features.shape[0]
        log_flat = log_features.reshape(n_samples, -1)  # 展平为每个样本一个向量
        metric_flat = metric_features.reshape(n_samples, -1)
        trace_flat = trace_features.reshape(n_samples, -1)
        
        print(f"展平后的日志特征形状: {log_flat.shape}")
        print(f"展平后的指标特征形状: {metric_flat.shape}")
        print(f"展平后的调用链特征形状: {trace_flat.shape}")
        
        # 拼接所有特征
        concatenated_features = np.hstack([log_flat, metric_flat, trace_flat])
        
        print(f"拼接后的特征形状: {concatenated_features.shape}")
        
        # 存储结果
        concat_features = {
            'log': log_flat.tolist(),
            'metric': metric_flat.tolist(),
            'trace': trace_flat.tolist(),
            'concatenated': concatenated_features.tolist(),
            'y': features['y']
        }
        
        concat_path = os.path.join(self.config["save_dir"], f"{self.config['dataset']}_concat_features.json")
        with open(concat_path, 'w') as f:
            json.dump(concat_features, f)
        
        print(f"拼接后的特征已保存到: {concat_path}")
        
        return concat_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多模态数据特征提取和融合工具")
    parser.add_argument("--dataset", type=str, required=True, help="数据集名称")
    parser.add_argument("--task", type=str, required=True, 
                        choices=["process_metrics", "process_service", "fusion"],
                        help="任务类型：process_metrics=处理指标数据, process_service=处理单个服务日志、trace数据, fusion=融合特征")
    parser.add_argument("--service", type=str, default=None, help="处理特定服务实例的日志和调用链数据 (仅在task=process_service时使用)")
    parser.add_argument("--discovered_services", type=str, default=None, 
                        help="自动发现的服务实例列表，以空格分隔 (仅在task=fusion时使用)")
    
    args = parser.parse_args()
    
    U.set_seed(2024)
    time_str = datetime.now().strftime("%Y年%m月%d日%H时%M分%S秒")
    
    # 获取配置
    config = CONFIG_DICT[args.dataset].copy()
    
    # 如果有自动发现的服务列表，更新配置
    if args.discovered_services and args.task == "fusion":
        discovered_services = args.discovered_services.split(" ")
        print(f"使用自动发现的服务列表: {discovered_services}")
        config["instances"] = args.discovered_services
    
    # 如果指定了特定服务
    if args.service:
        config["service_name"] = args.service
    else:
        # 默认选择第一个服务
        services = config["instances"].split(" ")
        config["service_name"] = services[0]
    
    # 创建处理器
    processor = FeatureProcessor(config, time_str)
    
    # 根据任务类型执行相应操作
    if args.task == "process_metrics":
        # 处理指标数据并自动拆分到服务目录
        processor.process_metrics()
    
    elif args.task == "process_service":
        # 处理特定服务的日志和调用链数据
        if args.service is None:
            print("错误：处理服务数据时必须指定--service参数")
            exit(1)
        
        # 设置服务名并处理数据
        processor.process_service(args.service)
    
    elif args.task == "fusion":
        # 融合特征
        processor.merge_features()  # 这会自动调用simple_concat_features和apply_feature_extraction

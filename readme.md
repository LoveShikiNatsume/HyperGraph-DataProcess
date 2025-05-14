# HyperGraph-DataProcess

一个用于处理多模态系统监控数据的特征提取和融合工具。本项目能够处理日志、指标和调用链数据，并将它们融合成统一的特征矩阵，为后续的故障诊断提供基础。

## 项目概述

本项目设计用于处理分布式系统中的多模态数据，特别是:

- **指标数据(Metric)**: 系统和服务的性能指标，如CPU、内存使用率等
- **日志数据(Log)**: 服务产生的文本日志记录
- **调用链数据(Trace)**: 服务间调用关系和延迟数据

通过处理和融合这些数据，最终生成特征矩阵，可用于故障检测、定位和根因分析。

## 项目结构

```
HyperGraph-DataProcess/
├── config.py                # 配置文件，包含各数据集的配置参数
├── main.py                  # 核心处理逻辑，数据处理的主要入口
├── process_all.py           # 批处理脚本，处理所有服务实例数据
├── utils.py                 # 工具函数
├── dataset/                 # 数据集处理类
│   ├── __init__.py          # 导出数据集类
│   ├── base_dataset.py      # 基础数据集类
│   ├── log.py               # 日志数据处理类
│   ├── metric.py            # 指标数据处理类
│   ├── trace.py             # 调用链数据处理类
│   └── drain3/              # 日志模板提取工具
├── model/                   # 编码器和融合模型
│   ├── encoder.py           # 各模态数据的编码器
│   └── fusion.py            # 特征融合模型
└── README.md                # 项目文档
```

## 数据流

1. **原始数据** → 从各系统收集的原始监控数据
2. **数据预处理** → 清洗、规范化和特征提取
3. **模态编码** → 使用特定编码器处理不同模态数据
4. **特征融合** → 将多模态特征融合为统一特征矩阵
5. **输出结果** → 保存处理结果供下游任务使用

## 配置文件说明

配置文件(`config.py`)中包含了针对不同数据集的详细配置，每个数据集配置包括:

- **数据路径**: 数据目录、服务实例目录、保存目录等
- **处理参数**: 采样间隔、时间窗口、工作线程数等
- **数据集信息**: 服务列表、故障类型、日期范围等
- **模型参数**: 编码器维度、注意力头数、层数等

示例配置:

```python
gaia = {
    "dataset": "gaia",
    "dataset_dir": "../datasets/train-ticket",       # 指标数据的根目录
    "service_dir": "../datasets/train-ticket/service", # 服务实例数据的根目录
    "save_dir": "../datasets/processed",             # 处理后数据保存目录
    # ... 更多配置参数 ...
}
```

## 使用方法

### 1. 使用批处理脚本处理所有服务数据

```bash
python process_all.py --dataset <dataset_name> [--mode all|log|metric|trace]
```

参数说明:
- `--dataset`: 要处理的数据集名称，必须是config.py中定义的数据集
- `--mode`: 处理模式，默认为"all"(全部模态)，可选"log"(仅日志)、"metric"(仅指标)、"trace"(仅调用链)

### 2. 直接使用main.py处理特定任务

```bash
python main.py --dataset <dataset_name> --task <task_name> [--service <service_name>] [--discovered_services <service_list>]
```

参数说明:
- `--dataset`: 要处理的数据集名称
- `--task`: 任务类型，可选值:
  - `process_metrics`: 处理指标数据并拆分到各服务目录
  - `process_service`: 处理特定服务的日志和调用链数据
  - `fusion`: 融合所有特征
- `--service`: (仅与task=process_service一起使用)指定要处理的服务名称
- `--discovered_services`: (仅与task=fusion一起使用)自动发现的服务实例列表

## 核心类和函数

### 1. FeatureProcessor

`FeatureProcessor`类(main.py)是整个数据处理的核心，主要方法:

- `process_metrics()`: 处理指标数据并拆分到各服务目录
- `process_service()`: 处理特定服务的日志和调用链数据
- `merge_features()`: 合并所有模态的特征
- `simple_concat_features()`: 简单拼接三种模态特征
- `apply_feature_extraction()`: 使用编码器提取更高级特征

### 2. 数据集类

所有数据集类都继承自`BaseDataset`(dataset/base_dataset.py):

- `LogDataset`: 处理日志数据，使用Drain算法提取模板并使用BERT编码
- `MetricDataset`: 处理指标数据，包括时间序列处理和特征提取
- `TraceDataset`: 处理调用链数据，提取调用关系和延迟特征

新增的服务路径处理方法:
- `set_service_paths()`: 为特定服务设置数据目录和保存目录
- `get_data_dir()`: 返回当前使用的数据目录

### 3. 编码器和融合模型

`model/`目录包含各种编码器和融合模型:

- `LogEncoder`: 日志数据编码器，基于Transformer结构
- `MetricEncoder`: 指标数据编码器，处理时间序列特征
- `TraceEncoder`: 调用链数据编码器，处理调用关系图
- `ConcatFusion`: 简单拼接融合模型
- `GatedFusion`: 门控融合模型
- `AdaFusion`: 自适应融合模型

## 数据处理流程

1. **服务实例发现**:
   - `process_all.py`自动扫描服务目录，发现所有服务实例文件夹
   - 或使用配置文件中预定义的服务列表

2. **指标数据处理**:
   - 加载全局指标数据
   - 通过Z-score归一化和差分处理时间序列
   - 拆分指标数据到各个服务目录

3. **服务数据处理**:
   - 为每个服务单独处理日志和调用链数据
   - 日志处理: 模板提取、BERT编码、特征加权
   - 调用链处理: 延迟序列提取、统计特征计算

4. **特征融合**:
   - 合并所有服务的特征矩阵
   - 提供两种融合方法:
     - 简单拼接: 直接拼接各模态特征
     - 编码器提取: 使用专门的编码器提取更高级表示

5. **结果保存**:
   - 中间结果: 保存各模态处理后的特征
   - 最终结果: 保存融合后的特征矩阵

## 输入数据要求

### 指标数据

期望的列: `timestamp`, `cmdb_id`, `kpi_name`, `value`

### 日志数据

期望的列: `timestamp`, `message`

### 调用链数据

期望的列: `timestamp`, `st_time`, `ed_time`, `parent_id`, `cmdb_id`

### 标记数据(Groundtruth)

期望的列: `st_time`, `ed_time`, `failure_type`, `root_cause`

## 输出文件

处理后的数据保存在配置中指定的`save_dir`目录:

- `{dataset}_metric_tmp.json`: 原始指标特征
- `service/{service}/{dataset}_log_tmp.json`: 各服务日志特征
- `service/{service}/{dataset}_trace_tmp.json`: 各服务调用链特征
- `service/{service}/{dataset}_metric_tmp.json`: 拆分后的服务指标特征
- `{dataset}_merged_features.json`: 合并后的特征
- `{dataset}_concat_features.json`: 简单拼接的特征
- `{dataset}_encoded_features.json`: 编码器提取的特征

## 扩展和自定义

### 添加新的数据集

1. 在`config.py`中添加新的数据集配置
2. 定义数据路径、服务列表、故障类型等参数

### 自定义特征提取

1. 修改相应的数据集类(`log.py`, `metric.py`, `trace.py`)
2. 重写`load()`方法实现自定义数据加载和处理逻辑

### 定制融合方法

1. 在`model/fusion.py`中添加新的融合模型
2. 修改`main.py`中的融合逻辑

## 代码示例

### 处理新服务的数据

```python
from main import FeatureProcessor
from config import CONFIG_DICT

# 加载配置
config = CONFIG_DICT["gaia"].copy()
config["service_name"] = "my_new_service"

# 创建处理器
processor = FeatureProcessor(config, "2023-01-01")

# 处理服务数据
processor.process_service("my_new_service")

# 融合特征
processor.merge_features()
```

## 常见问题

1. **找不到服务目录**
   - 确保配置中的`service_dir`路径正确
   - 检查目录权限和文件存在性

2. **处理时内存不足**
   - 调整批处理大小
   - 减少并行处理的工作线程数

3. **特征维度不匹配**
   - 确保所有服务都有相同的指标列表
   - 检查日志和调用链数据的完整性

## 项目开发者

如果您是项目开发者，以下信息会帮助您理解和修改代码:

### 关键设计决策

1. **中心化配置管理**:
   - 所有配置都集中在`config.py`
   - 其他模块通过依赖注入接收配置

2. **服务特定路径处理**:
   - 使用`set_service_paths`方法设置服务特定路径
   - 避免修改全局配置

3. **统一数据接口**:
   - 所有数据集类继承自`BaseDataset`
   - 标准化的`load()`和`save_to_tmp()`方法

4. **可扩展的特征融合**:
   - 支持多种特征融合方式
   - 编码器与融合模型分离

### 主要代码流程

1. **process_all.py**:
   - 发现服务实例
   - 按顺序处理指标、服务数据、融合特征

2. **main.py**:
   - 根据命令行参数执行不同任务
   - 处理指标、服务数据、融合特征

3. **dataset/\*.py**:
   - 加载和预处理原始数据
   - 提取和保存特征

4. **model/\*.py**:
   - 定义各种编码器和融合模型
   - 提供特征提取功能
```
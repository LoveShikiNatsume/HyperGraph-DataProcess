from .log import UnifiedLogDataset as LogDataset
from .metric import UnifiedMetricDataset as MetricDataset
from .trace import UnifiedTraceDataset as TraceDataset

# 简化的数据集映射
LOG_DATASET = {
    "aiops22": LogDataset,
    "platform": LogDataset,
    "gaia": LogDataset,
    "ob": LogDataset,
    "tt": LogDataset
}

METRIC_DATASET = {
    "aiops22": MetricDataset,
    "platform": MetricDataset,
    "gaia": MetricDataset,
    "ob": MetricDataset,
    "tt": MetricDataset
}

TRACE_DATASET = {
    "aiops22": TraceDataset,
    "platform": TraceDataset,
    "gaia": TraceDataset,
    "ob": TraceDataset,
    "tt": TraceDataset
}

# 直接导出统一的类，便于导入
__all__ = ['LogDataset', 'MetricDataset', 'TraceDataset', 
           'LOG_DATASET', 'METRIC_DATASET', 'TRACE_DATASET']

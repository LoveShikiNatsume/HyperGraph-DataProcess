gaia = {
    # dataset
    "dataset": "gaia",
    "dataset_dir": "../datasets/train-ticket",       # 指标数据的根目录
    "service_dir": "../datasets/train-ticket/service", # 服务实例数据的根目录
    "save_dir": "../datasets/processed",             # 处理后数据保存目录
    "log_name": "experiment",
    "use_tmp": True,
    "num_workers": 26,
    "sample_interval": 60,
    "service_name": None,  # 将在运行时设置
    "drain_config": {
        "drain_save_path": "../datasets/processed/drain.bin",  # 将在运行时根据服务名动态调整
        "drain_config_path": "dataset/drain3/drain.ini",
    },
    "bert_config": {
        "tokenizer_path": "cache/bert-base-uncased",
        "model_path": "cache/bert-base-uncased",
    },
    # 数据集特定参数
    "dates": ["2025-03-03 18_57_36", "2025-03-04 13_49_54", "2025-03-04 16_07_12",
             "2025-03-04 18_15_07", "2025-03-04 20_17_18", "2025-03-05 09_29_09",
             "2025-03-05 11_35_31", "2025-03-05 13_40_20", "2025-03-05 16_42_21", 
             "2025-03-05 21_03_15"],
    "anomaly_dict": {
        "[Fault.CPU]": "cpu",
        "[Fault.MEMORY]": "memory",
        "[Fault.NETWORK_LOSS]": "api",
        "[Fault.NETWORK_LATENCY]": "api",
        "[Node]": "node",
    },
    "file_patterns": {
        "log": "log",
        "trace": "trace",
        "metric": ["cpu", "memory", "rx", "tx"],
        "groundtruth": "groundtruth"
    },
    # label info
    "failures": "cpu memory api node",
    "services": "logservice1 logservice2 mobservice1 mobservice2 redisservice1 redisservice2 dbservice1 dbservice2 webservice1 webservice2",
    "instances": "logservice1 logservice2 mobservice1 mobservice2 redisservice1 redisservice2 dbservice1 dbservice2 webservice1 webservice2",
    "label_type": "failure_type",
    "num_class": 4,
    # cuda
    "use_cuda": True,
    "gpu": 0,
    # model parameters for feature extraction
    "max_len": 512,
    "d_model": 768,
    "nhead": 8,
    "d_ff": 256,
    "layer_num": 2,
    "dropout": 0.3,
}

aiops22 = {
    # dataset
    "dataset": "aiops22",
    "dataset_dir": "../datasets/aiops2022-pre",
    "service_dir": "../datasets/aiops2022-pre/service", # 服务实例数据的根目录
    "save_dir": "data/aiops22",
    "log_name": "experiment",
    "use_tmp": True,
    "num_workers": 10,
    "sample_interval": 60,
    "service_name": None,  # 将在运行时设置
    "drain_config": {
        "drain_save_path": "data/aiops22/drain.bin",  # 将在运行时根据服务名动态调整
        "drain_config_path": "dataset/drain3/drain.ini",
    },
    "bert_config": {
        "tokenizer_path": "cache/bert-base-uncased",
        "model_path": "cache/bert-base-uncased",
    },
    # 数据集特定参数
    "dates": [
        "2022-05-01",
        "2022-05-03",
        "2022-05-05",
        "2022-05-07",
        "2022-05-09",
    ],
    "anomaly_dict": {
        "k8s容器网络延迟": "network",
        "k8s容器写io负载": "io",
        "k8s容器读io负载": "io",
        "k8s容器cpu负载": "cpu",
        "k8s容器网络资源包重复发送": "network",
        "k8s容器进程中止": "process",
        "k8s容器网络丢包": "network",
        "k8s容器内存负载": "memory",
        "k8s容器网络资源包损坏": "network",
    },
    "file_patterns": {
        "log": "-log-service",
        "trace": "trace_jaeger-span",
        "metric": "kpi_container",
        "groundtruth": "groundtruth-"
    },
    # label info
    "failures": "cpu io memory network process",
    "services": "adservice cartservice checkoutservice currencyservice emailservice frontend paymentservice productcatalogservice recommendationservice shippingservice",
    "instances": "adservice-0 adservice-1 adservice-2 adservice2-0 cartservice-0 cartservice-1 cartservice-2 cartservice2-0 checkoutservice-0 checkoutservice-1 checkoutservice-2 checkoutservice2-0 currencyservice-0 currencyservice-1 currencyservice-2 currencyservice2-0 emailservice-0 emailservice-1 emailservice-2 emailservice2-0 frontend-0 frontend-1 frontend-2 frontend2-0 paymentservice-0 paymentservice-1 paymentservice-2 paymentservice2-0 productcatalogservice-0 productcatalogservice-1 productcatalogservice-2 productcatalogservice2-0 recommendationservice-0 recommendationservice-1 recommendationservice-2 recommendationservice2-0 shippingservice-0 shippingservice-1 shippingservice-2 shippingservice2-0",
    "label_type": "failure_type",
    "num_class": 5,
    # cuda
    "use_cuda": True,
    "gpu": 0,
    # model parameters for feature extraction
    "max_len": 512,
    "d_model": 768,
    "nhead": 8,
    "d_ff": 256,
    "layer_num": 1,
    "dropout": 0.35,
}

ob = {
    # dataset
    "dataset": "ob",
    "dataset_dir": "../datasets/new_platform",
    "service_dir": "../datasets/new_platform/service", # 服务实例数据的根目录
    "save_dir": "data/ob",
    "log_name": "experiment",
    "use_tmp": True,
    "num_workers": 10,
    "sample_interval": 60,
    "service_name": None,  # 将在运行时设置
    "drain_config": {
        "drain_save_path": "data/ob/drain.bin",  # 将在运行时根据服务名动态调整
        "drain_config_path": "dataset/drain3/drain.ini",
    },
    "bert_config": {
        "tokenizer_path": "cache/bert-base-uncased",
        "model_path": "cache/bert-base-uncased",
    },
    # 数据集特定参数
    "dates": [
        "2024-03-22",
        "2024-03-23",
        "2024-03-24",
    ],
    "anomaly_dict": {
        "cpu anomaly": "cpu",
        "http/grpc request abscence": "http/grpc",
        "http/grpc requestdelay": "http/grpc",
        "memory overload": "memory",
        "network delay": "network",
        "network loss": "network",
        "pod anomaly": "pod_failure",
    },
    "file_patterns": {
        "log": "log",
        "trace": "trace",
        "metric": "kpi_container",
        "groundtruth": "ground_truth"
    },
    # label info
    "failures": "cpu http/grpc memory network pod_failure",
    "services": "cartservice checkoutservice currencyservice emailservice frontend paymentservice productcatalogservice recommendationservice shippingservice",
    "instances": "cartservice checkoutservice currencyservice emailservice frontend paymentservice productcatalogservice recommendationservice shippingservice",
    "label_type": "failure_type",
    "num_class": 5,
    # cuda
    "use_cuda": True,
    "gpu": 0,
    # model parameters for feature extraction
    "max_len": 512,
    "d_model": 768,
    "nhead": 8,
    "d_ff": 256,
    "layer_num": 1,
    "dropout": 0.1,
}

tt = {
    # dataset
    "dataset": "tt",
    "dataset_dir": "../datasets/train-ticket-new",      # 指标数据的根目录
    "service_dir": "../datasets/train-ticket-new/service", # 服务实例数据的根目录
    "save_dir": "../datasets/processed-tt",            # 处理后数据保存目录
    "log_name": "experiment",
    "use_tmp": True,
    "num_workers": 26,
    "sample_interval": 60,
    "service_name": None,  # 将在运行时设置
    "drain_config": {
        "drain_save_path": "../datasets/processed-tt/drain.bin",  # 将在运行时根据服务名动态调整
        "drain_config_path": "dataset/drain3/drain.ini",
    },
    "bert_config": {
        "tokenizer_path": "cache/bert-base-uncased",
        "model_path": "cache/bert-base-uncased",
    },
    # 数据集特定参数
    "dates": ["2025-03-03 18_57_36", "2025-03-04 13_49_54", "2025-03-04 16_07_12",
             "2025-03-04 18_15_07", "2025-03-04 20_17_18", "2025-03-05 09_29_09",
             "2025-03-05 11_35_31", "2025-03-05 13_40_20", "2025-03-05 16_42_21", 
             "2025-03-05 21_03_15"],
    "anomaly_dict": {
        "[Fault.CPU]": "cpu",
        "[Fault.MEMORY]": "memory",
        "[Fault.NETWORK_LOSS]": "api",
        "[Fault.NETWORK_LATENCY]": "api",
        "[Node]": "node",
    },
    "file_patterns": {
        "log": "log",
        "trace": "trace",
        "metric": ["cpu", "memory", "rx", "tx"],
        "groundtruth": "groundtruth"
    },
    # label info
    "failures": "cpu memory api node",
    "services": "ts-assurance-service ts-auth-service ts-basic-service ts-config-service ts-contacts-service ts-food-service ts-order-service ts-preserve-service ts-price-service ts-route-service ts-seat-service ts-station-service ts-ticketinfo-service ts-train-service ts-travel-service ts-user-service ts-verification-code-service",
    "instances": "ts-assurance-service ts-auth-service ts-basic-service ts-config-service ts-contacts-service ts-food-service ts-order-service ts-preserve-service ts-price-service ts-route-service ts-seat-service ts-station-service ts-ticketinfo-service ts-train-service ts-travel-service ts-user-service ts-verification-code-service",
    "label_type": "failure_type",
    "num_class": 4,
    # cuda
    "use_cuda": True,
    "gpu": 0,
    # model parameters for feature extraction
    "max_len": 512,
    "d_model": 768,
    "nhead": 8,
    "d_ff": 256,
    "layer_num": 2,
    "dropout": 0.3,
}

CONFIG_DICT = {"gaia": gaia, "aiops22": aiops22, "ob": ob, "tt": tt}

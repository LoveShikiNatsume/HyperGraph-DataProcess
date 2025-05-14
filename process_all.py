import os
import argparse
from config import CONFIG_DICT
import subprocess
import time

def discover_service_instances(service_dir):
    """
    自动发现服务目录下的服务实例文件夹
    
    Args:
        service_dir: 服务数据的根目录
        
    Returns:
        服务实例名称列表
    """
    if not os.path.exists(service_dir):
        raise FileNotFoundError(f"服务目录不存在: {service_dir}")
    
    print(f"正在从目录 {service_dir} 发现服务实例...")
    
    # 获取service_dir下的所有子文件夹作为服务实例
    services = [d for d in os.listdir(service_dir) if os.path.isdir(os.path.join(service_dir, d))]
    
    if not services:
        raise ValueError(f"服务目录中没有发现任何服务实例: {service_dir}")
    
    print(f"发现了 {len(services)} 个服务实例: {', '.join(services[:5])}{'...' if len(services) > 5 else ''}")
    
    return services

def process_all_services(dataset, mode):
    """处理指定数据集的所有服务"""
    config = CONFIG_DICT[dataset]
    service_dir = config.get("service_dir")
    
    if not service_dir:
        raise ValueError(f"配置中没有指定service_dir: {dataset}")
    
    # 自动发现服务实例
    try:
        services = discover_service_instances(service_dir)
        print(f"在{service_dir}中发现了 {len(services)} 个服务实例")
    except Exception as e:
        print(f"自动发现服务实例失败: {str(e)}")
        print(f"将使用配置文件中定义的服务列表...")
        services = config["instances"].split(" ")
    
    print(f"准备处理数据集 {dataset} 的 {len(services)} 个服务: {', '.join(services)}")
    
    # 首先处理指标数据 - 只需要运行一次
    if mode in ["all", "metric"]:
        print(f"\n处理指标数据...")
        cmd = ["python", "main.py", "--dataset", dataset, "--task", "process_metrics"]
        subprocess.run(cmd, check=True)
    
    # 然后为每个服务处理日志和调用链数据
    if mode in ["all", "log", "trace"]:
        for i, service in enumerate(services):
            print(f"\n[{i+1}/{len(services)}] 处理服务 {service}...")
            
            # 处理服务数据（日志和调用链）
            cmd = ["python", "main.py", "--dataset", dataset, "--task", "process_service", 
                   "--service", service]
            subprocess.run(cmd, check=True)
    
    # 融合所有特征前，确保配置文件中的实例列表是最新的
    # 这是为了确保融合时能正确处理所有发现的服务实例
    if services != config["instances"].split(" "):
        print("\n更新配置文件中的服务实例列表...")
        # 这里我们不实际修改配置文件，而是在运行融合任务时传递发现的服务列表
        services_str = " ".join(services)
        cmd = ["python", "main.py", "--dataset", dataset, "--task", "fusion", 
               "--discovered_services", services_str]
        subprocess.run(cmd, check=True)
    else:
        # 使用默认配置进行融合
        print("\n融合所有特征...")
        cmd = ["python", "main.py", "--dataset", dataset, "--task", "fusion"]
        subprocess.run(cmd, check=True)
    
    print(f"\n所有处理完成! 特征文件保存在: {config['save_dir']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批处理所有服务的数据")
    parser.add_argument("--dataset", type=str, required=True, help="数据集名称")
    parser.add_argument("--mode", type=str, default="all", 
                        choices=["all", "log", "metric", "trace"],
                        help="处理模式：all=所有, log=仅日志, metric=仅指标, trace=仅调用链")
    
    args = parser.parse_args()
    
    start_time = time.time()
    process_all_services(args.dataset, args.mode)
    end_time = time.time()
    
    print(f"总处理时间: {(end_time - start_time)/60:.2f} 分钟")

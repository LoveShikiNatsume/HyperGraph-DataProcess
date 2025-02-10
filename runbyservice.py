import os

def modify_gaia_config(filepath, dataset_dir, save_dir, drain_save_dir):
    # 读取 config.py 文件
    with open(filepath, "r") as file:
        lines = file.readlines()

    # 初始化新的内容
    new_lines = []
    in_gaia_section = False

    # 遍历原文件的每一行
    for line in lines:
        stripped_line = line.strip()

        # 检查是否进入 gaia 部分
        if stripped_line.startswith("gaia = {"):
            in_gaia_section = True

        # 检查是否离开 gaia 部分
        if in_gaia_section and stripped_line.startswith("aiops22 = {"):
            in_gaia_section = False

        # 修改 gaia 部分的指定配置
        if in_gaia_section:
            if stripped_line.startswith('"dataset_dir":'):
                line = f'    "dataset_dir": "{dataset_dir}",\n'
            elif stripped_line.startswith('"save_dir":'):
                line = f'    "save_dir": "{save_dir}",\n'
            elif stripped_line.startswith('"drain_save_path":'):
                line = f'    "drain_save_path": "{drain_save_dir}",\n'

        # 添加修改后的行到新的内容
        new_lines.append(line)

    # 将修改后的内容写回文件
    with open(filepath, "w") as file:
        file.writelines(new_lines)

def process_data_and_train(base_dir, config_path):
    # 遍历所有服务实例文件夹
    for service_folder in sorted(os.listdir(base_dir)):

        if service_folder not in ['logservice1', 'logservice2']:
            continue

        service_path = os.path.join(base_dir, service_folder)
        if not os.path.isdir(service_path) or service_folder.startswith("2021-"):
            continue

        # 避免使用缓存 Log
        tmp_path = service_path + '/gaia_log_tmp.json'
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        drain_path = service_path + "/drain.bin"
        if os.path.exists(drain_path):
            os.remove(drain_path)

        # 避免使用缓存 Trace
        # tmp_path = service_path + '/gaia_trace_tmp.json'
        # if os.path.exists(tmp_path):
        #     os.remove(tmp_path)

        # 修改 config.py 文件
        modify_gaia_config(
            filepath=config_path,
            dataset_dir=service_path,
            save_dir=service_path,
            drain_save_dir=service_path + "/drain.bin"
        )

        print(f"Processing service: {service_folder}")
        # 运行训练脚本
        os.system("python main.py --dataset gaia")

# 主程序调用
base_directory = "../datasets/service/"  # 数据集根目录
config_file_path = "./config.py"  # config.py 的路径

# 遍历所有服务文件夹并修改配置和训练
process_data_and_train(base_directory, config_file_path)

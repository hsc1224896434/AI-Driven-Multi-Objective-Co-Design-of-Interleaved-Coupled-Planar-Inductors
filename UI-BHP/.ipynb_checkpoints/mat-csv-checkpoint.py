import h5py
import pandas as pd
import os

# 设置MAT文件路径和输出目录
mat_file_path = r"/root/Data/3C90/3C90_TX-25-15-10_Data1_Cycle.mat"
output_dir = r'/root/Data/3C90/csv'

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 打开MAT文件
with h5py.File(mat_file_path, 'r') as file:
    # 访问Data组
    data_group = file['Data']

    # 定义要导出的数据集名称
    datasets_to_export = [
        'Voltage', 'Current', 'Sampling_Time', 'Temperature_command',
        'Hdc_command', 'DutyP_command', 'DutyN_command', 'Frequency_command', 'Flux_command'
    ]

    # 遍历数据集名称，读取并保存为CSV
    for dataset_name in datasets_to_export:
        # 构建完整的数据集路径
        full_dataset_path = f'Data/{dataset_name}'

        # 检查数据集是否存在于Data组中
        if dataset_name in data_group:
            # 读取数据集
            data = data_group[dataset_name][:]
            # 将数据转换为DataFrame
            df = pd.DataFrame(data)
            
            # 转置DataFrame
            df_transposed = df.T

            # 构建CSV文件路径
            csv_file_path = os.path.join(output_dir, f'{dataset_name}.csv')
            # 保存转置后的DataFrame为CSV文件
            df_transposed.to_csv(csv_file_path, index=False, header=False)
            print(f'{dataset_name} has been transposed and saved as {csv_file_path}')
        else:
            print(f'{dataset_name} does not exist in the Data group.')
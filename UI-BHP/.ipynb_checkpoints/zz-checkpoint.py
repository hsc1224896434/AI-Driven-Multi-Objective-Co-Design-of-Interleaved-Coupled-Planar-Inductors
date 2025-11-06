import pandas as pd
import os

# 设置CSV文件所在的目录
csv_dir = r'/root/Data/3C92/csv'

# 确保CSV目录存在
if not os.path.exists(csv_dir):
    print(f"The directory {csv_dir} does not exist.")
else:
    # 遍历目录中的所有CSV文件
    for filename in os.listdir(csv_dir):
        if filename.endswith('.csv'):
            # 构建完整的文件路径
            csv_file_path = os.path.join(csv_dir, filename)

            # 读取CSV文件
            df = pd.read_csv(csv_file_path, header=None)

            # 转置数据
            df_transposed = df.T

            # 保存转置后的数据到原始CSV文件路径
            df_transposed.to_csv(csv_file_path, index=False, header=False)
            print(f'{filename} has been transposed and overwritten.')
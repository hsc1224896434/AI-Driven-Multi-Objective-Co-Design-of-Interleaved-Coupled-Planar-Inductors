import h5py
import pandas as pd

# 定义文件路径
file_path = r'F:\Coreloss\Database 15 Material\Single Cycle\77'

try:
    # 尝试打开 HDF5 文件
    with h5py.File(file_path, 'r') as f:
        # 打印文件中的所有数据集和组
        print("文件中的数据集和组：")
        for key in f.keys():
            print(key)

        # 检查是否存在名为 'data' 的数据集
        if 'data' in f:
            # 获取数据
            data = f['data'][:]  # 将 HDF5 数据集转换为 NumPy 数组
            # 将数据转换为 DataFrame
            df = pd.DataFrame(data)
            # 保存为 CSV 文件，添加扩展名 .csv
            df.to_csv('77_0014_Data1_Combined.csv', index=False)
            print('数据已成功保存为 77_0014_Data1_Combined.csv')
        else:
            print('未找到名为 data 的变量')
except FileNotFoundError:
    print(f"文件未找到: {file_path}")
except PermissionError:
    print(f"没有足够的权限打开文件: {file_path}，请检查文件权限。")
except Exception as e:
    print(f"处理文件时出现其他错误: {e}")
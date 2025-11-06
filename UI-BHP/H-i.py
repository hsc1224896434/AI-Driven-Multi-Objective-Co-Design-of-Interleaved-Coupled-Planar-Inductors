import pandas as pd

# 读取CSV文件
input_file_path = r"磁场强度H.csv"
output_file_path = r"电流I.csv"

# 加载CSV文件
data = pd.read_csv(input_file_path, header=None)

# 定义要乘以的常数
scaling_factor = 0.008597  # 你可以修改为你想要的常数

# 将DataFrame中的所有数据乘以常数
scaled_data = data * scaling_factor

# 将结果保存为新的CSV文件
scaled_data.to_csv(output_file_path, index=False, float_format='%.8f')

print("Scaled data saved to:", output_file_path)

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:05:11 2024

@author: kim
"""

import pandas as pd
import numpy as np


def calculate_derivative(sequence, e_t):
    derivative = []
    for i in range(1, len(sequence)):
        dy = sequence[i] - sequence[i-1]  # 计算相邻元素的差异
        dx = e_t  # 使用传入的 e_t 作为相邻元素之间的距离
        derivative.append(dy / dx)  # 将差异除以距离得到导数值
    return np.array(derivative)

def Calculate_U_from_B(B_raw, e_t, N_test, A_e):
    # 计算电压 U_t
    U_integral_t = calculate_derivative(B_raw, e_t)* N_test * A_e
    U_t = U_integral_t - (max(U_integral_t) + min(U_integral_t)) / 2
    return U_t

# 读取 B_t 数据的.csv 文件
B_data = pd.read_csv(r"磁通密度B.csv", header=None, dtype=float)

# 读取频率数据的.csv 文件
freq_data = pd.read_csv(r"频率f.csv", header=None)

# 确保频率数据与 B_t 数据的行数一致
if len(freq_data) != len(B_data):
    raise ValueError("The number of rows in the frequency data file does not match the number of rows in the B_t data file.")

# 初始化参数
N_test = 7
OD = 25 * 1e-3
ID = 15 * 1e-3
H = 10 * 1e-3
A_e = ((OD - ID) / 2) * H

# 计算对应的 U_t 数据
num = B_data.shape[0]
U_array = np.zeros([num,1023])

for i, (B_row, freq_row) in enumerate(zip(B_data.values, freq_data.values)):
    freq = freq_row[0]  # 从频率数据中提取频率值
    e_t = 1/(freq * 1024) # 计算 e_t 值
    U_t = Calculate_U_from_B(B_row, e_t, N_test, A_e)
    U_array[i] = U_t

# 保存为新的.csv 文件
np.savetxt(r"电压U.csv", U_array, delimiter=',', fmt='%.8f')

print('Finish')




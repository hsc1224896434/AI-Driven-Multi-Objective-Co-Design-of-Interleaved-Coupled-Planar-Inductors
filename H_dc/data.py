import numpy as np
import math
import os
from scipy.interpolate import interp1d
import h5py

def Calculate_B_H_P(U_raw, I_raw, R_sense, OD, ID, H, N_exciting, N_test, mu_0, e_t, Freq):
    A_e = 4e-05
    l_e = 0.0620
    T = 1 / Freq
    sample = min(math.ceil(1 / (Freq * e_t)), len(U_raw) - 1)  # 确保 sample 不超过 U_raw 的长度

    # 计算直流部分
    U_raw_integal, I_raw_integal, P_t = 0, 0, 0
    for k in range(sample):
        U_raw_integal += ((U_raw[k] + U_raw[k + 1]) / 2) * e_t
        I_raw_integal += ((I_raw[k] + I_raw[k + 1]) / 2) * e_t
    U_dc, I_dc = U_raw_integal / T, I_raw_integal / T

    # 一个周期的 U,I,H,B
    U_t = U_raw[:sample] - U_dc
    I_t = I_raw[:sample] - I_dc
    H_t = (I_t * N_exciting) / l_e
    H_t = H_t.flatten()
    B_integal_t = (np.cumsum(U_t) * e_t) / (N_test * A_e)
    B_t = B_integal_t - (max(B_integal_t) + min(B_integal_t)) / 2  # 修正了拼写错误
    H_max, H_min = max(H_t), min(H_t)
    B_max, B_min = max(B_t), min(B_t)
    # 计算 P
    P_t = 0
    for k in range(sample):
        P_t += U_t[k] * I_t[k] * e_t
    P_t = P_t / T
    P_cv = P_t / (A_e * l_e)
    mu = ((B_max - B_min) / (H_max - H_min)) / mu_0
    return B_t, H_t, P_cv

def resample_sequence(sequence, sample_num):
    seq_len = len(sequence)

    if seq_len > sample_num:
        # 下采样
        step = seq_len // sample_num
        resampled_sequence = sequence[::step][:sample_num]
    else:
        # 上采样并使用线性插值
        x_old = np.arange(seq_len)
        x_new = np.linspace(0, seq_len - 1, sample_num)
        interpolator = interp1d(x_old, sequence, kind='linear')
        resampled_sequence = interpolator(x_new)

    return resampled_sequence

if __name__ == "__main__":
    # 材料文件夹
    folder_path = r"F:\Coreloss\Database 15 Material\Single Cycle\77"
    mat_file_name = "77_0014_Data1_Cycle.mat"
    mat_file_path = os.path.join(folder_path, mat_file_name)

    # 参数设置
    R_sense = 1
    OD = 22 * 1e-3
    ID = 14 * 1e-3
    H = 6.4 * 1e-3
    N_exciting = 7
    N_test = 7
    mu_0 = 4 * 3.1415 * 1e-7
    # 采样点个数
    sample_num = 1024
    # 精度设置，例如保留 10 位小数
    decimal_places = 10

    try:
        # 检查文件是否存在
        if not os.path.exists(mat_file_path):
            print(f"文件不存在: {mat_file_path}")
            exit(1)

        with h5py.File(mat_file_path, 'r') as f:
            # 检查MAT文件的结构
            print("MAT文件结构:", list(f.keys()))

            # 假设数据存储在 'Data' 中
            data_group = f['Data']
            print("Data组的结构:", list(data_group.keys()))

            # 检查每个字段的形状
            voltage_data = np.array(data_group['Voltage'][:])  # 提取 Voltage 数据
            voltage_data = voltage_data.T
            print("Voltage 数据形状:", voltage_data.shape)

            current_data = np.array(data_group['Current'][:])  # 提取 Current 数据
            current_data = current_data.T
            print("Current 数据形状:", current_data.shape)

            sampling_time_data = np.array(data_group['Sampling_Time'][:])  # 提取 Sampling_Time 数据
            sampling_time_data = sampling_time_data.T
            print("Sampling_Time 数据形状:", sampling_time_data.shape)

            temperature_data = np.array(data_group['Temperature_command'][:])  # 提取 Temperature_command 数据
            temperature_data = temperature_data.T
            print("Temperature_command 数据形状:", temperature_data.shape)

            frequency_data = np.array(data_group['Frequency_command'][:])  # 提取 Frequency_command 数据
            frequency_data = frequency_data .T
            print("Frequency_command 数据形状:", frequency_data.shape)

            Hdc_data = np.array(data_group['Hdc_command'][:])[0, :]  # 提取第一行数据
            print("Hdc_command 数据形状:", Hdc_data.shape)

            # 确保所有数据的行数相同
            num_rows_voltage = voltage_data.shape[0]
            num_rows_current = current_data.shape[0]
            num_rows_sampling_time = sampling_time_data.shape[0]
            num_rows_temperature = temperature_data.shape[0]
            num_rows_frequency = frequency_data.shape[0]

            # 找出最小的行数
            num_rows = min(num_rows_voltage, num_rows_current, num_rows_sampling_time, num_rows_temperature,
                           num_rows_frequency)

            # 裁剪数据以确保行数相同
            voltage_data = voltage_data[:num_rows, :]
            current_data = current_data[:num_rows, :]
            sampling_time_data = sampling_time_data[:num_rows, :]
            temperature_data = temperature_data[:num_rows, :]
            frequency_data = frequency_data[:num_rows, :]

        B_array = []
        H_array = []
        T_array = []
        F_array = []
        P_array = []

        for i in range(num_rows):
            Freq = frequency_data[i, 0]
            Temperature = temperature_data[i, 0]
            e_t = sampling_time_data[i, 0]

            U_raw = voltage_data[i, :].astype(float)
            I_raw = current_data[i, :].astype(float)
            B_t, H_t, P = Calculate_B_H_P(U_raw, I_raw, R_sense, OD, ID, H, N_exciting, N_test, mu_0, e_t, Freq)
            B_t = resample_sequence(B_t, sample_num)
            H_t = resample_sequence(H_t, sample_num)

            # 将数据四舍五入到指定的小数位数
            B_t = np.around(B_t, decimals=decimal_places)
            H_t = np.around(H_t, decimals=decimal_places)
            P = np.around(P, decimals=decimal_places)
            Freq = np.around(Freq, decimals=decimal_places)
            Temperature = np.around(Temperature, decimals=decimal_places)

            B_array.append(B_t)
            H_array.append(H_t)
            T_array.append(Temperature)
            F_array.append(Freq)
            P_array.append(P)

        # 转换为 NumPy 数组
        B_array = np.array(B_array)
        H_array = np.array(H_array)
        T_array = np.array(T_array).reshape(-1, 1)
        F_array = np.array(F_array).reshape(-1, 1)
        P_array = np.array(P_array).reshape(-1, 1)

        # 确保输出的 B_array 和 H_array 的形状为 (30248, 1024)
        if B_array.shape != (30248, 1024):
            raise ValueError(f"B_array 的形状不正确，期望 (30248, 1024)，实际 {B_array.shape}")

        # 确保输出的 T_array、F_array、P_array 的形状为 (30248, 1)
        if T_array.shape != (30248, 1):
            raise ValueError(f"T_array 的形状不正确，期望 (30248, 1)，实际 {T_array.shape}")

        if F_array.shape != (30248, 1):
            raise ValueError(f"F_array 的形状不正确，期望 (30248, 1)，实际 {F_array.shape}")

        if P_array.shape != (30248, 1):
            raise ValueError(f"P_array 的形状不正确，期望 (30248, 1)，实际 {P_array.shape}")

        # 保存结果
        np.savetxt(os.path.join(folder_path, 'B.csv'), B_array, delimiter=',', fmt=f'%.{decimal_places}f')
        np.savetxt(os.path.join(folder_path, 'H.csv'), H_array, delimiter=',', fmt=f'%.{decimal_places}f')
        np.savetxt(os.path.join(folder_path, 'T.csv'), T_array, delimiter=',', fmt=f'%.{decimal_places}f')
        np.savetxt(os.path.join(folder_path, 'F.csv'), F_array, delimiter=',', fmt=f'%.{decimal_places}f')
        np.savetxt(os.path.join(folder_path, 'P.csv'), P_array, delimiter=',', fmt=f'%.{decimal_places}f')

        np.savetxt(os.path.join(folder_path, 'Hdc.csv'), Hdc_data, delimiter=',', fmt=f'%.{decimal_places}f')

        print('finish')

    except FileNotFoundError:
        print(f"未找到文件: {mat_file_path}，请检查路径和文件名。")
    except KeyError as e:
        print(f"文件中未找到指定的字段: {str(e)}，请检查字段名称。")
    except ValueError as e:
        print(f"数据形状错误: {e}")
    except Exception as e:
        print(f"加载文件时出现错误: {e}")
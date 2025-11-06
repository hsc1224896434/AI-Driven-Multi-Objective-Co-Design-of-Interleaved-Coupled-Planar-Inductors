import pandas as pd
import numpy as np
import math
import os
from scipy.interpolate import interp1d
import h5py


def Calculate_B_H_P(U_raw, I_raw, R_sense, OD, ID, H, N_exciting, N_test, mu_0, e_t, Freq):
    A_e = 1.3e-04
    l_e = 0.0351
    T = 1 / Freq

    # 确保 U_raw 是数组并且长度足够
    if isinstance(U_raw, (int, float, np.float64)):
        U_raw = np.array([U_raw])
    if len(U_raw) < 2:
        print("U_raw 长度不足，无法进行计算。")
        return np.array([]), np.array([]), 0

    sample = min(math.ceil(1 / (Freq * e_t)), len(U_raw) - 1)
    # 确保 sample 至少为 1
    sample = max(sample, 1)

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
    B_t = B_integal_t - (max(B_integal_t) + min(B_integal_t)) / 2
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
        step = seq_len // sample_num
        resampled_sequence = sequence[::step][:sample_num]
    else:
        x_old = np.arange(seq_len)
        x_new = np.linspace(0, seq_len - 1, sample_num)
        interpolator = interp1d(x_old, sequence, kind='linear')
        resampled_sequence = interpolator(x_new)
    return resampled_sequence


if __name__ == "__main__":
    folder_path = r"F:\Coreloss\Database 15 Material\Single Cycle\77"
    mat_file_name = "77_0014_Data1_Cycle.mat"
    mat_file_path = os.path.join(folder_path, mat_file_name)

    R_sense = 1
    OD = 22 * 1e-3
    ID = 14 * 1e-3
    H = 6.4 * 1e-3
    N_exciting = 1
    N_test = 1
    mu_0 = 4 * 3.1415 * 1e-7
    sample_num = 1024
    decimal_places = 10

    try:
        with h5py.File(mat_file_path, 'r') as f:
            struct_data = f['Data']  # 假设结构体名称为 'Data'

            def get_dataset(struct, field_name, file):
                ref = struct[field_name][0, 0]
                if isinstance(ref, h5py.Reference):
                    data = np.array(file[ref])
                    if data.ndim == 0:
                        data = data.reshape(1, 1)
                    elif data.ndim == 1:
                        data = data.reshape(-1, 1) if data.size > 1 else data.reshape(1, -1)
                    return data
                return np.array([ref])

            voltage_data = get_dataset(struct_data, 'Voltage', f)
            current_data = get_dataset(struct_data, 'Current', f)
            e_t_data = get_dataset(struct_data, 'Sampling_Time', f)
            temperature_data = get_dataset(struct_data, 'Temperature_command', f)
            frequency_data = get_dataset(struct_data, 'Frequency_command', f)

    except FileNotFoundError:
        print(f"未找到文件: {mat_file_path}，请检查路径和文件名。")
        exit(1)
    except KeyError as e:
        print(f"文件中未找到指定的字段: {str(e)}，请检查字段名称。")
        exit(1)
    except Exception as e:
        print(f"加载文件时出现错误: {e}")
        exit(1)

    num_rows = len(frequency_data)
    if (len(temperature_data) != num_rows or
            voltage_data.shape[0] != num_rows or
            current_data.shape[0] != num_rows or
            len(e_t_data) != num_rows):
        raise ValueError("The number of rows in the data must be the same.")

    B_array = np.empty((0, sample_num))
    H_array = np.empty((0, sample_num))
    T_array = np.empty((0, 1))
    F_array = np.empty((0, 1))
    P_array = np.empty((0, 1))

    for i in range(num_rows):
        Freq = frequency_data[i, 0] if frequency_data.ndim == 2 else frequency_data[i]
        Temperature = temperature_data[i, 0] if temperature_data.ndim == 2 else temperature_data[i]
        e_t = e_t_data[i, 0] if e_t_data.ndim == 2 else e_t_data[i]
        U_raw = voltage_data[i, :].astype(float) if voltage_data.ndim == 2 else voltage_data[i].astype(float)
        I_raw = current_data[i, :].astype(float) if current_data.ndim == 2 else current_data[i].astype(float)

        # 确保 U_raw 和 I_raw 是一维数组
        if isinstance(U_raw, (int, float, np.float64)):
            U_raw = np.array([U_raw])
        if isinstance(I_raw, (int, float, np.float64)):
            I_raw = np.array([I_raw])

        B_t, H_t, P = Calculate_B_H_P(U_raw, I_raw, R_sense, OD, ID, H, N_exciting, N_test, mu_0, e_t, Freq)
        if len(B_t) > 0:
            B_t = resample_sequence(B_t, sample_num)
            H_t = resample_sequence(H_t, sample_num)

            B_t = np.around(B_t, decimals=decimal_places)
            H_t = np.around(H_t, decimals=decimal_places)
            P = np.around(P, decimals=decimal_places)
            Freq = np.around(Freq, decimals=decimal_places)
            Temperature = np.around(Temperature, decimals=decimal_places)

            B_array = np.vstack((B_array, B_t))
            H_array = np.vstack((H_array, H_t))
            T_array = np.vstack((T_array, Temperature))
            F_array = np.vstack((F_array, Freq))
            P_array = np.vstack((P_array, P))

    np.savetxt(os.path.join(folder_path, 'B.csv'), B_array, delimiter=',', fmt=f'%.{decimal_places}f')
    np.savetxt(os.path.join(folder_path, 'H.csv'), H_array, delimiter=',', fmt=f'%.{decimal_places}f')
    np.savetxt(os.path.join(folder_path, 'T.csv'), T_array, delimiter=',', fmt=f'%.{decimal_places}f')
    np.savetxt(os.path.join(folder_path, 'F.csv'), F_array, delimiter=',', fmt=f'%.{decimal_places}f')
    np.savetxt(os.path.join(folder_path, 'P.csv'), P_array, delimiter=',', fmt=f'%.{decimal_places}f')

    print('finish')
import pandas as pd
import numpy as np
import math
from scipy.interpolate import interp1d
import os


def Calculate_B_H_P(U_raw, I_raw, R_sense, OD, ID, H, N_exciting, N_test, mu_0, e_t, Freq):
    A_e = ((OD - ID) / 2) * H
    l_e = 3.1415 * ((OD + ID) / 2)
    T = 1 / Freq
    sample = math.ceil(1 / (Freq * e_t))
    sample = min(sample, len(U_raw) - 1, len(I_raw) - 1)
    print(f"U_raw length: {len(U_raw)}, sample: {sample}")

    U_raw_integal, I_raw_integal, P_t = 0, 0, 0
    for k in range(sample):
        U_raw_integal += ((U_raw[k] + U_raw[k + 1]) / 2) * e_t
        I_raw_integal += ((I_raw[k] + I_raw[k + 1]) / 2) * e_t
    U_dc, I_dc = U_raw_integal / T, I_raw_integal / T
    U_t = U_raw[:sample] - U_dc
    I_t = I_raw[:sample] - I_dc
    H_t = (I_t * N_exciting) / l_e
    H_t = H_t.to_numpy().flatten()  # 修改这一行
    B_integal_t = (np.cumsum(U_t) * e_t) / (N_test * A_e)
    B_t = B_integal_t - (max(B_integal_t) + min(B_integal_t)) / 2
    H_max, H_min = max(H_t), min(H_t)
    B_max, B_min = max(B_t), min(B_t)
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
    folder_path = "/root/Data/3C90/csv/"  # 假设您的CSV文件都存放在这个文件夹下
    R_sense = 1
    OD = 22 * 1e-3
    ID = 14 * 1e-3
    H = 6.4 * 1e-3
    N_exciting = 4
    N_test = 4
    mu_0 = 4 * 3.1415 * 1e-7
    e_t = 4 * 1e-9
    sample_num = 1024

    # 读取电压和电流数据
    U_data = pd.read_csv(os.path.join(folder_path, 'Voltage.csv'))
    I_data = pd.read_csv(os.path.join(folder_path, 'Current.csv'))
    U_raw = U_data.iloc[:, 0].astype(float)
    I_raw = I_data.iloc[:, 0].astype(float)

    # 读取频率和温度数据
    Freq_data = pd.read_csv(os.path.join(folder_path, 'Frequency_command.csv'))
    Temperature_data = pd.read_csv(os.path.join(folder_path, 'Temperature_command.csv'))
    Freq = Freq_data.iloc[0, 0]  # 假设频率是固定的，取第一个值
    Temperature = Temperature_data.iloc[0, 0]  # 假设温度是固定的，取第一个值

    # 计算B, H, P
    B_t, H_t, P = Calculate_B_H_P(U_raw, I_raw, R_sense, OD, ID, H, N_exciting, N_test, mu_0, e_t, Freq)

    # 重采样B和H
    B_t = resample_sequence(B_t, sample_num)
    H_t = resample_sequence(H_t, sample_num)

    # 保存结果
    np.savetxt('/root/Data/3C90/csv/B.csv', B_t, delimiter=',')
    np.savetxt('/root/Data/3C90/csv/H.csv', H_t, delimiter=',')
    np.savetxt('/root/Data/3C90/csv/P.csv', [P], delimiter=',')  # P是单个值，所以需要用列表包裹

    print('finish')
import pandas as pd
import numpy as np
import math
import os
from scipy.interpolate import interp1d

def Calculate_B_H_P(U_raw, I_raw, R_sense, OD, ID, H, N_exciting, N_test, mu_0, e_t, Freq):
    A_e = ((OD-ID)/2)*H
    l_e = 3.1415*((OD+ID)/2)
    T = 1/Freq
    sample = math.ceil(1/(Freq*e_t))

    # 计算直流部分
    U_raw_integal, I_raw_integal, P_t = 0, 0, 0
    for k in range(sample):
        U_raw_integal += ((U_raw[k] + U_raw[k + 1]) / 2) * e_t
        I_raw_integal += ((I_raw[k] + I_raw[k + 1]) / 2) * e_t
    U_dc, I_dc = U_raw_integal/T, I_raw_integal/T
    # 一个周期的U,I,H,B
    U_t = U_raw[:sample] - U_dc
    I_t = I_raw[:sample] - I_dc
    H_t = (I_t * N_exciting) / l_e
    H_t = H_t.flatten()
    B_integal_t = (np.cumsum(U_t) * e_t)/(N_test*A_e)
    B_t = B_integal_t - (max(B_integal_t) + min(B_integal_t))/2
    H_max, H_min = max(H_t), min(H_t)
    B_max, B_min = max(B_t), min(B_t)
    # 计算P
    P_t = 0
    for k in range(sample):
        P_t += U_t[k]*I_t[k]*e_t
    P_t = P_t/T
    P_cv = P_t/(A_e*l_e)
    mu = ((B_max-B_min)/(H_max-H_min))/mu_0
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
    folder_path = "C:/Users/CaptainZ/Desktop/3E6"
    # 参数设置
    R_sense = 1
    OD = 22 * 1e-3
    ID = 14 * 1e-3
    H = 6.4 * 1e-3
    N_exciting = 4
    N_test = 4
    mu_0 = 4 * 3.1415 * 1e-7
    e_t = 4 * 1e-9
    # 采样点个数
    sample_num = 1024

    B_array = np.empty((0, sample_num))
    H_array = np.empty((0, sample_num))
    T_array = np.empty((0, 1))
    F_array = np.empty((0, 1))
    P_array = np.empty((0, 1))
    sub_folder_names = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    for sub_folder_name in sub_folder_names:
        Temperature = int(sub_folder_name)
        folder_temp_path = folder_path + "/" + sub_folder_name
        Temp_Freq_folder_names = [f for f in os.listdir(folder_temp_path) if
                                  os.path.isdir(os.path.join(folder_temp_path, f))]
        for Temp_Freq_folder_name in Temp_Freq_folder_names:
            parts = Temp_Freq_folder_name.split("K")
            Freq = int(parts[0]) * 1e3
            csv_path = folder_temp_path + "/" + Temp_Freq_folder_name
            csv_names = os.listdir(csv_path)
            num = int(len(csv_names) / 2)
            for i in range(num):
                U_raw_path = csv_path + '/' + csv_names[i]
                I_raw_path = csv_path + '/' + csv_names[i + num]
                U_data = pd.read_csv(U_raw_path)
                I_data = pd.read_csv(I_raw_path)
                U_raw = U_data.iloc[4:,0].astype(float)
                I_raw = I_data.iloc[4:, 0].astype(float)
                U_raw, I_raw = U_raw.values, I_raw.values
                B_t, H_t, P = Calculate_B_H_P(U_raw, I_raw, R_sense, OD, ID, H, N_exciting, N_test, mu_0, e_t, Freq)
                B_t = resample_sequence(B_t, sample_num)
                H_t = resample_sequence(H_t, sample_num)
                B_array = np.vstack((B_array, B_t))
                H_array = np.vstack((H_array, H_t))
                T_array = np.vstack((T_array, Temperature))
                F_array = np.vstack((F_array, Freq))
                P_array = np.vstack((P_array, P))

    np.savetxt('B.csv', B_array, delimiter=',')
    np.savetxt('H.csv', H_array, delimiter=',')
    np.savetxt('T.csv', T_array, delimiter=',')
    np.savetxt('F.csv', F_array, delimiter=',')
    np.savetxt('P.csv', P_array, delimiter=',')

    print('finish')

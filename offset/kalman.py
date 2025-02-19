import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 卡尔曼滤波函数
def apply_optimized_kalman(signal, t):
    n = len(t)
    output = np.zeros(n)
    A = 1  # 状态转移矩阵
    H = 1  # 观测矩阵
    x_est = signal[0]  # 初始状态估计
    P_est = 1  # 初始估计协方差
    q_min = 0.0001
    q_max = 0.001
    r_min = 0.3
    r_max = 1.6
    output[0] = signal[0]

    for i in range(1, n):
        # Q R 动态调整
        signal_change = abs(signal[i] - signal[i-1])
        Q = q_min + (q_max - q_min) * (signal_change / max(signal))
        R = r_min + (r_max - r_min) * (signal_change / max(signal))

        # 预测
        x_pred = A * x_est
        P_pred = A * P_est * A + Q

        # 更新
        K = P_pred * H / (H * P_pred * H + R)
        x_est = x_pred + K * (signal[i] - H * x_pred)
        P_est = (1 - K * H) * P_pred

        output[i] = x_est

    return output

# 读取CSV文件
def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df

# 可视化数据
def plot_data(original_data_1, original_data_2, filtered_data_1, filtered_data_2, pic_path):
    plt.figure(figsize=(10, 6))

    # 绘制原始数据
    plt.plot(original_data_1, 'b--', label='OT 1 (--)', alpha=0.3)
    plt.plot(original_data_2, 'r--', label='Pred 2 (--)', alpha=0.3)

    # 绘制卡尔曼滤波后的数据
    plt.plot(filtered_data_1, 'b-', label='Filtered OT 1 (blue)',alpha=0.7)
    plt.plot(filtered_data_2, 'r-', label='Filtered Pred 2 (red)',alpha=0.7)

    plt.title('original data and kalman filter ans')
    plt.xlabel('time step')
    plt.ylabel('time diff/ms')
    plt.legend()
    plt.grid(True)
    plt.savefig(pic_path, dpi=300)
    plt.close()

# 计算滑动均值并调整信号
def smooth_and_flip_signal(signal, window_size=10):
    # 计算滑动均值
    rolling_mean = np.convolve(signal, np.ones(window_size) / window_size, mode='same')
    
    # 调整信号：原始信号 - 均值 + (1 - 均值)
    adjusted_signal = signal - rolling_mean + (1 - rolling_mean)
    
    # 翻转信号
    flipped_signal = adjusted_signal[::-1]
    
    return flipped_signal, rolling_mean

# 主函数
def show_kalman_filter(file_path, pic_path):
    # 读取CSV文件
    df = read_csv(file_path)
    # df = df[:30000]
    df = df[:6000]
    # df = df[-100:]
    # df = df[70000:]
    data_1 = df['real'].values
    # data_2 = df['forecast'].values*1.04
    data_2 = df['forecast'].values
    time = np.arange(len(data_1))  # 时间轴

    # 应用优化后的卡尔曼滤波
    filtered_data_1 = apply_optimized_kalman(data_1, time)
    filtered_data_2 = apply_optimized_kalman(data_2, time)
    diff_data = filtered_data_1 - filtered_data_2
    # diff_data = data_1 - data_2
    plt.plot(diff_data)
    plt.title('Difference Between Filtered Signals')
    plt.xlabel('time step/s')
    plt.ylabel('Difference/ms')
    plt.legend()
    plt.grid(True)
    plt.savefig(pic_path+"diff_ms_small.png", dpi=300)
    plt.close()
    # 可视化数据
    plot_data(data_1, data_2, filtered_data_1, filtered_data_2, pic_path=pic_path+'kalman_filterd.png')
    # df2 = read_csv(file_path_2)
    # data_1=df2['OT'].values
    # filtered_data_1 = apply_optimized_kalman(data_1, time)
    # # plot_data(data_1, data_2, filtered_data_1, filtered_data_2)
    # # data_1,_ = smooth_and_flip_signal(data_1)
    # # filtered_data_1 = apply_optimized_kalman(data_1, time)
    # data_1=1-data_1
    # filtered_data_1=1-filtered_data_1
    # plot_data(data_1, data_2, filtered_data_1, filtered_data_2, pic_path='kalman_final_ans2_end2.png')

# 运行主程序
if __name__ == "__main__":
    # 请将文件路径替换为你的CSV文件路径
    # file_path = './results/P-ForecastResults.csv'
    file_path = '/mnt/e/timer/offset/org_and_ans_ms.csv'
    # file_path_2 = '/mnt/e/timer/offset/timestamp.csv'
    # file_path_2 = 'O.csv'
    show_kalman_filter(file_path)

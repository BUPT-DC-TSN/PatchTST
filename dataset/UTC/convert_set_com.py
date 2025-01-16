import numpy as np
import pandas as pd

def calculate_allan_variance(data, tau):
    """
    计算 Allan 方差
    :param data: [UTC–UTC(k)] 数据，单位：纳秒
    :param tau: 采样间隔，单位：秒
    :return: Allan 方差
    """
    N = len(data)
    if N < 2:
        raise ValueError("数据点不足，无法计算 Allan 方差")
    
    # 计算差分
    diff = np.diff(data)
    
    # 计算平方和
    squared_diff_sum = np.sum(diff**2)
    
    # 计算 Allan 方差
    allan_variance = squared_diff_sum / (2 * (N - 1) * tau**2)
    
    return allan_variance

def generate_set_com(data, tau):
    """
    生成 Set-com
    :param data: 包含 [UTC–UTC(k)] 和其他信息的数据框
    :param tau: 采样间隔，单位：秒
    :return: Set-com 数据框
    """
    # 筛选 Group A 的数据
    group_a_data = data[data['Group'] == 'A'].copy()
    
    # 计算每个时钟的 Allan 方差
    allan_variances = {}
    for (lab_id, clock), group in group_a_data.groupby(['Lab_ID', 'Clock']):
        utc_diff = group['[UTC-UTC(k)]/ns'].values
        try:
            avar = calculate_allan_variance(utc_diff, tau)
            allan_variances[(lab_id, clock)] = avar
        except ValueError as e:
            print(f"实验室 {lab_id} 的时钟 {clock} 数据不足：{e}")
            allan_variances[(lab_id, clock)] = np.nan  # 如果数据不足，设为 NaN
    
    # 将 Allan 方差添加到数据框中
    group_a_data['Allan Variance'] = group_a_data.apply(
        lambda row: allan_variances.get((row['Lab_ID'], row['Clock']), np.nan), axis=1
    )
    
    # 删除 Allan 方差为 NaN 的数据
    group_a_data = group_a_data.dropna(subset=['Allan Variance'])
    
    # 处理 Allan 方差为 0 的情况
    epsilon = 1e-12  # 极小值
    group_a_data['Allan Variance'] = group_a_data['Allan Variance'].replace(0, epsilon)
    
    # 计算权重（Allan 方差的倒数）
    group_a_data['Weight'] = 1 / group_a_data['Allan Variance']
    
    # 归一化权重
    total_weight = group_a_data['Weight'].sum()
    group_a_data['Weight'] = group_a_data['Weight'] / total_weight
    
    # 按 MJD 分组，计算加权平均的 [UTC–UTC(k)]
    set_com = group_a_data.groupby('MJD', group_keys=False).apply(
        lambda x: np.sum(x['[UTC-UTC(k)]/ns'] * x['Weight'])
    ).reset_index(name='[UTC-UTC(k)]/ns')
    
    return set_com

# 读取数据
data = pd.read_csv('merged_data.csv')

# 假设采样间隔为 1 天（86400 秒）
tau = 86400  # 单位：秒

# 生成 Set-com
set_com = generate_set_com(data, tau)

# 保存 Set-com 到文件
set_com.to_csv('set_com.csv', index=False)

print("Set-com 已生成并保存为 set_com.csv")
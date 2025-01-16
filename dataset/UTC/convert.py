import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 将MJD转换为日期时间
def mjd_to_datetime(mjd):
    """
    将MJD（Modified Julian Date）转换为Python的datetime对象。
    MJD的起点是1858年11月17日。
    """
    return datetime(1858, 11, 17) + timedelta(days=mjd)

# 读取drift.txt文件并解析
def parse_drift_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    current_mjds = None

    for line in lines:
        if line.startswith("X:\\"):  # 检测到新段落的开头
            current_mjds = None  # 重置当前MJD
        elif "LAB." in line:  # 检测到表头行
            # 提取MJD，并过滤掉无效字符（例如 '.'）
            current_mjds = [mjd for mjd in line.split()[2:] if mjd.replace('.', '').isdigit()]
            current_mjds = [int(mjd) for mjd in current_mjds]  # 将MJD转换为整数
        elif current_mjds and line.strip():  # 处理数据行
            parts = line.split()
            lab = parts[0]
            clock = ''.join(parts[1:3])
            drifts = parts[3:]
            for i in range(len(drifts)):
                if i < len(current_mjds):  # 确保数据与MJD对应
                    data.append([lab, clock, current_mjds[i], drifts[i]])

    # 转换为DataFrame
    df_drift = pd.DataFrame(data, columns=['Laboratory', 'Clock', 'MJD', 'Frequency Drift/ns'])
    
    # 将缺失值（如*********）替换为NaN
    df_drift['Frequency Drift/ns'] = df_drift['Frequency Drift/ns'].replace('*********', np.nan)
    
    # 将MJD和Frequency Drift/ns列转换为数值类型
    df_drift['MJD'] = pd.to_numeric(df_drift['MJD'], errors='coerce')  # 将无效值转换为NaN
    df_drift['Frequency Drift/ns'] = pd.to_numeric(df_drift['Frequency Drift/ns'], errors='coerce')
    
    # 删除MJD为NaN的行
    df_drift = df_drift.dropna(subset=['MJD'])
    
    # 将MJD转换为日期时间
    df_drift['MJD'] = df_drift['MJD'].apply(mjd_to_datetime)
    
    # 检查并处理重复的MJD值
    df_drift = df_drift.groupby(['Laboratory', 'Clock']).apply(
        lambda group: group.drop_duplicates(subset=['MJD'])
                      .set_index('MJD')
                      .resample('5D')
                      .asfreq()  # 保持原始频率，不进行插值
                      .ffill()  # 前向填充其他列
                      .reset_index()
    ).reset_index(drop=True)
    
    # 将日期时间转换回MJD
    df_drift['MJD'] = (df_drift['MJD'] - datetime(1858, 11, 17)).dt.days
    
    return df_drift

# 读取circularT.txt文件并解析
def parse_circular_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    current_mjds = None

    for line in lines:
        if line.startswith("Date"):  # 检测到新段落的开头
            continue
        elif "MJD" in line:  # 检测到MJD行
            current_mjds = line.split()[1:-3]  # 提取MJD
            current_mjds = [int(mjd) for mjd in current_mjds]  # 将MJD转换为整数
        elif "Laboratory k" in line:  # 检测到表头行
            continue  # 跳过表头
        elif current_mjds and line.strip():  # 处理数据行
            pattern = re.compile(r"([A-Z]+)\s+\([^)]+\)\s+([\d\.\-\s]+)")
            match = pattern.match(line)
            if match:
                lab = match.group(1)  # 站点名称
                # 处理数值数据，将 '-' 替换为 NaN
                values = []
                for value in match.group(2).split():
                    if value == '-':
                        values.append(float('nan'))  # 将 '-' 替换为 NaN
                    else:
                        values.append(float(value))  # 正常转换为浮点数
                utc_diff = values[:-3]
                uA = values[-3]

            for i in range(len(utc_diff)):
                if i < len(current_mjds):  # 确保数据与日期和MJD对应
                    data.append([
                        lab, current_mjds[i], utc_diff[i], uA
                    ])

    # 转换为DataFrame
    df_circular = pd.DataFrame(data, columns=[
        'Laboratory', 'MJD', '[UTC-UTC(k)]/ns', 'uA'
    ])
    
    # 将缺失值（如-）替换为NaN
    df_circular['[UTC-UTC(k)]/ns'] = df_circular['[UTC-UTC(k)]/ns'].replace('-', np.nan)
    df_circular['uA'] = df_circular['uA'].replace('-', np.nan)
    
    # 将MJD、OT和u列转换为数值类型
    df_circular['MJD'] = pd.to_numeric(df_circular['MJD'], errors='coerce')  # 将无效值转换为NaN
    df_circular['[UTC-UTC(k)]/ns'] = pd.to_numeric(df_circular['[UTC-UTC(k)]/ns'], errors='coerce')
    df_circular['uA'] = pd.to_numeric(df_circular['uA'], errors='coerce')
    
    # 删除MJD为NaN的行
    df_circular = df_circular.dropna(subset=['MJD'])

    return df_circular

# 对缺失值进行线性插值
def interpolate_missing_values(df, column):
    df[column] = df.groupby(['Laboratory', 'Clock'])[column].apply(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    ).reset_index(level=[0, 1], drop=True)
    return df

# 主程序
def main():
    # 解析drift.txt和circularT.txt文件
    df_drift = parse_drift_file('/mnt/e/timer/PatchTST/dataset/UTC/drift.txt')
    df_circular = parse_circular_file('/mnt/e/timer/PatchTST/dataset/UTC/CircularT.txt')

    # 合并两个数据集
    df_merged = pd.merge(df_circular, df_drift, on=['Laboratory', 'MJD'], how='outer')

    # 对缺失值进行线性插值
    df_merged = interpolate_missing_values(df_merged, 'Frequency Drift/ns')
    df_merged = interpolate_missing_values(df_merged, '[UTC-UTC(k)]/ns')
    df_merged = interpolate_missing_values(df_merged, 'uA')

    # 将MJD列转换为int类型
    df_merged['MJD'] = df_merged['MJD'].astype(int)

    # 删除OT为空的行
    df_merged = df_merged.dropna(subset=['[UTC-UTC(k)]/ns'])

    # 按实验室名称分配编号（从1开始）
    df_merged['Lab_ID'] = df_merged.groupby('Laboratory').ngroup() + 1

    # 根据 [UTC-UTC(k)] 的绝对值划分 Group A~E
    bins = [0, 3, 10, 50, 100, float('inf')]
    labels = ['A', 'B', 'C', 'D', 'E']
    df_merged['Group'] = pd.cut(df_merged['[UTC-UTC(k)]/ns'].abs(), bins=bins, labels=labels, right=False)

    # df_merged = df_merged[(df_merged['MJD'] <= 60600)] # 参照论文采390天的数据

    # 保存为CSV文件
    df_merged.to_csv('merged_data.csv', index=False)
    print("数据已成功处理并保存为 merged_data.csv")


    # 统计每个Group中有多少个Laboratory
    group_lab_count = df_merged.groupby('Group', observed=False)['Laboratory'].nunique()
    print("每个Group中的Laboratory数量：")
    print(group_lab_count)

    # 输出每个Group中的所有Laboratory名称
    group_lab_names = df_merged.groupby('Group', observed=False)['Laboratory'].unique()
    print("\n每个Group中的Laboratory名称：")
    for group, labs in group_lab_names.items():
        print(f"Group {group}: {', '.join(labs)}\n")


# 运行主程序
if __name__ == "__main__":
    main()
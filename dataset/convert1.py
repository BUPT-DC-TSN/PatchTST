import pandas as pd
import argparse
import re
import os

def convert(data_dir):
    # 读取文件change_timestamps.csv
    df = pd.read_csv(os.path.join(data_dir, "change_timestamps.csv"))
    # 计算奇数行与偶数行的差值
    df['TimeAll'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H-%M-%S.%f')
    data = []
    for i in range(0, len(df)-1, 2):
        time_diff = df.iloc[i+1]['TimeAll'] - df.iloc[i]['TimeAll']
        timestamp = df.iloc[i]['TimeAll']
        data.append([timestamp, time_diff])
    # 将数据保存为 CSV 文件
    time_diff_df = pd.DataFrame(data, columns=["timestamp", "diff"])
    time_diff_df.to_csv(os.path.join(data_dir, "time_diff.csv"), index=False)
    
    time_diff_dict = {}
    for _, row in time_diff_df.iterrows():
        timestamp = str(row['timestamp']).split('.')[0]  # 只取整数部分
        time_diff_dict[timestamp] = row['diff']
    
    # 打开 log 文件并读取所有行
    log_file_path = os.path.join(data_dir, 'logs.txt')
    with open(log_file_path, "r") as file:
        log_lines = file.readlines()
    
    # 用来存储匹配的数据
    matching_data = []
    
    # 正则表达式，用于提取日志中的信息
    log_pattern = re.compile(r'(?P<timestamp>\d+\.\d+) ptp4l\[\d+\.\d+\]: master offset\s+(?P<master_offset>-?\d+)\s+s2 freq\s+(?P<freq>-?\+?\d+)\s+path delay\s+(?P<path_delay>\d+)')
    
    # 遍历日志文件中的每一行，提取数据
    for line in log_lines:
        match = log_pattern.search(line)
        if match:
            timestamp = match.group("timestamp")
            timestamp_match = match.group("timestamp").split('.')[0]  # 只保留整数部分的时间戳
            master_offset = int(match.group("master_offset"))
            freq = int(match.group("freq"))
            path_delay = int(match.group("path_delay"))
            
            # 从 time_diff 字典中查找对应的 diff
            if timestamp_match in time_diff_dict:
                diff = time_diff_dict[timestamp_match]
                matching_data.append([timestamp, diff, master_offset, freq, path_delay])
    
    output_df = pd.DataFrame(matching_data, columns=["time", "target", "master_offset", "freq", "path_delay"])
    
    # 不存一下会有精度问题
    output_df.to_csv(os.path.join(data_dir, 'tmp.csv'), index=False)
    data = os.path.join(data_dir, 'tmp.csv')
    df = pd.read_csv(data)
    
    df['date'] = pd.to_datetime(df['time'], unit='s')
    df['OT'] = df['target']
    df = df.drop(columns=['time', 'target'])
    
    # 打印处理后的数据
    print(df)
    
    # 计算均值和方差
    mean_freq = df['freq'].mean()
    variance_freq = df['freq'].std()
    
    mean_path_delay = df['path_delay'].mean()
    variance_path_delay = df['path_delay'].std()
    
    mean_master_offset = df['master_offset'].mean()
    variance_master_offset = df['master_offset'].std()
    
    print(f"Mean of freq: {mean_freq}, Variance of freq: {variance_freq}")
    print(f"Mean of path_delay: {mean_path_delay}, Variance of path_delay: {variance_path_delay}")
    print(f"Mean of master_offset: {mean_master_offset}, Variance of master_offset: {variance_master_offset}")
    
    df.to_csv(os.path.join(data_dir, 'timestamp.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='give me data path')
    parser.add_argument("--data_date_path", type=str, default='0112')
    args = parser.parse_args()
    convert(args.data_date_path)
import pandas as pd
from bisect import bisect_left

def process_data(txt_path, csv_path, output_path):
    # 解析攻击日志
    attacks = []
    with open(txt_path) as f:
        current_attack = {}
        for line in f:
            line = line.strip()
            if not line:
                continue
          
            if '攻击开始' in line:
                parts = line.split(',')
                current_attack = {
                    'start': int(parts[1].split('：')[1].replace(' ms', '')) / 1000,
                    'mode': int(parts[2].split('：')[1]),
                    'end': None
                }
                # 处理可能的未闭合攻击
                if '持续时间' in line:
                    duration = int(parts[3].split('：')[1].replace('秒', ''))
                    current_attack['end'] = current_attack['start'] + duration
            elif '攻击结束' in line:
                end_time = int(line.split('：')[1].replace(' ms', '')) / 1000
                current_attack['end'] = end_time
                attacks.append(current_attack)
                current_attack = {}
  
    # 生成时间区间结构
    starts = []
    ends = []
    modes = []
    for attack in sorted(attacks, key=lambda x: x['start']):
        starts.append(attack['start'])
        ends.append(attack['end'])
        modes.append(attack['mode'])
  
    # 处理CSV数据
    df = pd.read_csv(csv_path)
    df['timestamp'] = df['timestamp'].astype(float)
  
    # 二分查找优化
    def get_attack_type(t):
        idx = bisect_left(starts, t) - 1
        if idx >= 0 and starts[idx] <= t <= ends[idx]:
            return modes[idx]
        return 0
  
    df['attack_type'] = df['timestamp'].apply(get_attack_type)
  
    # 保存结果
    df.to_csv(output_path, index=False)
    print(f"处理完成，结果已保存至 {output_path}")

if __name__ == "__main__":
  
    process_data('2.14attackL.txt', 'cleaned_data.csv', 'fixed_data.csv')
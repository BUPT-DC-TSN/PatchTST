import pandas as pd
from bisect import bisect_left
import re

def process_data(txt_path, csv_path, output_path):
    # 解析攻击日志
    attacks = []
    current_attack = {}
  
    # 使用正则表达式优化解析
    pattern = re.compile(r'$$(攻击开始|攻击结束)$$\s+(.*)')
    param_pattern = re.compile(r'(\w+)[:：](\S+)')
  
    with open(txt_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
          
            # 解析事件类型和参数
            match = pattern.match(line)
            if not match:
                continue
              
            event_type, params_str = match.groups()
            params = {}
            for k, v in param_pattern.findall(params_str):
                params[k.strip()] = v.strip()
          
            # 处理攻击开始事件
            if event_type == '攻击开始':
                current_attack = {
                    'start': int(params['时间戳'].replace('ms', '')) / 1000,
                    'mode': int(params['模式']),
                    'target': int(params['目标值'].replace('ns', '')),
                    'end': None
                }
          
            # 处理攻击结束事件
            elif event_type == '攻击结束' and current_attack:
                current_attack['end'] = int(params['时间戳'].replace('ms', '')) / 1000
                attacks.append(current_attack)
                current_attack = {}

    # 生成时间区间结构（优化排序和校验）
    attacks.sort(key=lambda x: x['start'])
    starts = [a['start'] for a in attacks]
    ends = [a['end'] for a in attacks]
    modes = [a['mode'] for a in attacks]
    targets = [a['target'] for a in attacks]

    # 处理CSV数据（优化内存使用）
    df = pd.read_csv(csv_path, dtype={'timestamp': float})
  
    # 优化二分查找逻辑
    def find_attack(t):
        idx = bisect_left(starts, t) - 1
        if idx >= 0 and starts[idx] <= t <= ends[idx]:
            return (modes[idx], targets[idx])
        return (0, None)
  
    # 批量处理提升性能
    results = pd.DataFrame(
        df['timestamp'].apply(find_attack).tolist(),
        columns=['attack_type', 'attack_target']
    )
  
    # 合并结果
    df = pd.concat([df, results], axis=1)
  
    # 生成attack_log_value（使用矢量化操作提升性能）
    df['attack_log_value'] = df['attack_target'].where(
        df['attack_type'] == 3, 
        df['attack_value']
    )
  
    # 清理中间列
    df.drop(columns=['attack_target'], inplace=True)
  
    # 处理缺失值
    df.dropna(subset=['attack_log_value'], inplace=True)
  
    # 类型转换
    df['attack_type'] = df['attack_type'].astype(int)
    df['attack_log_value'] = df['attack_log_value'].astype(int)
  
    # 保存结果（优化内存使用）
    df.to_csv(output_path, index=False)
    print(f"数据处理完成，结果保存至 {output_path}")

if __name__ == "__main__":
    process_data('2.23log.txt', '2.23.csv', 'fixed_data_23.csv')
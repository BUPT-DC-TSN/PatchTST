import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from collections import defaultdict
import random
# 读取数据
df = pd.read_csv('full_predictions_20250221_192813.csv')
# df = pd.read_csv('full_predictions_with_value.csv')

# 筛选出攻击发生时的数据
attacked_df = df[df['is_attack'] == 1].copy()

# 计算分类准确率
attacked_df['correct'] = attacked_df['predicted_attack_type'] == attacked_df['attack_type']

# 原始攻击类型准确率
type_accuracy = attacked_df.groupby('attack_type')['correct'].mean().reset_index()
print("攻击类型准确率：")
print(type_accuracy.to_string(index=False))

# 新增：按attack_value四分位数分组计算准确率
attacked_df['value_quantile'] = pd.qcut(attacked_df['attack_value'], 
                                      q=4, 
                                      duplicates='drop')  # 处理重复分位值
quantile_accuracy = attacked_df.groupby('value_quantile')['correct'].agg(
    ['mean', 'count']
).reset_index()
quantile_accuracy.columns = ['attack_value区间', '平均准确率', '样本数量']

print("\n按攻击值区间的准确率：")
print(quantile_accuracy.to_string(index=False))

# 计算数值误差
error_metrics = attacked_df.groupby('attack_type').apply(
    lambda x: pd.Series({
        'MAE': mean_absolute_error(x['attack_value'], x['predicted_attack_value']),
        'RMSE': np.sqrt(mean_squared_error(x['attack_value'], x['predicted_attack_value']))
    })
).reset_index()
print("\n攻击数值误差指标：")
print(error_metrics.to_string(index=False))

# 寻找连续攻击段落
is_attack = df['is_attack'] == 1
changes = is_attack.ne(is_attack.shift()).cumsum()
attack_blocks = [group for _, group in df.groupby(changes) if group['is_attack'].all()]

if not attack_blocks:
    print("\n未找到连续攻击段落")
else:
    # 初始化统计数据结构
    step_counts = defaultdict(list)
    correct_counts = defaultdict(int)
    total_steps = defaultdict(int)
    split_accuracy = defaultdict(lambda: {'high': {'correct':0, 'total':0}, 
                                        'low': {'correct':0, 'total':0}})

    # 遍历所有攻击段落
    for block in attack_blocks:
        attack_type = block['attack_type'].iloc[0]
        same_mask = block['predicted_attack_type'] == block['attack_type']
        
        # 需求1: 计算首次匹配步数
        first_match_pos = next((i for i, v in enumerate(same_mask) if v), len(same_mask))
        step_counts[attack_type].append(first_match_pos)
        
        # 需求2: 统计正确次数和总步数
        correct = sum(same_mask)
        total = len(block)
        correct_counts[attack_type] += correct
        total_steps[attack_type] += total
        
        # 需求3: 按条件分组统计
        if attack_type in [1, 2]:
            # 类型1/2按最终值分组
            last_value = block['attack_value'].iloc[-1]
            key = 'high' if last_value > 4000000 else 'low'
        elif attack_type == 3:
            # 类型3按均值乘2分组
            avg_value = block['attack_value'].mean()
            multiplied_avg = avg_value * 2
            key = 'high' if multiplied_avg > 4000000 else 'low'
        else:
            continue  # 其他类型不处理
        
        split_accuracy[attack_type][key]['correct'] += correct
        split_accuracy[attack_type][key]['total'] += total

    # 输出需求1结果
    print("\n各攻击类型首次匹配前的平均步数：")
    for atype in sorted(step_counts):
        avg = np.mean(step_counts[atype])
        print(f"类型 {atype}: {avg:.2f} 步")

    # 输出需求2结果
    print("\n各攻击类型段落内准确率：")
    for atype in sorted(correct_counts):
        acc = correct_counts[atype] / total_steps[atype]
        print(f"类型 {atype}: {acc:.2%} ({correct_counts[atype]}/{total_steps[atype]})")

print("\n攻击类型按条件分组准确率：")
for atype in sorted(split_accuracy.keys()):
    # 确定条件描述
    if atype in [1, 2]:
        high_cond = ">4M（最终值）"
        low_cond = "≤4M（最终值）"
    elif atype == 3:
        high_cond = ">4M（平均×2）"
        low_cond = "≤4M（平均×2）"
    else:
        continue
    
    high_correct = split_accuracy[atype]['high']['correct']
    high_total = split_accuracy[atype]['high']['total']
    low_correct = split_accuracy[atype]['low']['correct']
    low_total = split_accuracy[atype]['low']['total']
    
    high_acc = high_correct / high_total if high_total > 0 else None
    low_acc = low_correct / low_total if low_total > 0 else None
    
    high_acc_str = f"{high_acc:.2%}" if high_acc is not None else "N/A"
    low_acc_str = f"{low_acc:.2%}" if low_acc is not None else "N/A"
    
    print(f"类型 {atype}（{high_cond}）: {high_acc_str} ({high_total}步)")
    print(f"类型 {atype}（{low_cond}）: {low_acc_str} ({low_total}步)")

    # 原代码的绘图部分
    attack_block = attack_blocks[random.randint(0,100)]
    plt.figure(figsize=(12, 6))
    x = attack_block.index
    plt.plot(x, attack_block['attack_value'], 
             label=f'True ({attack_block["attack_type"].iloc[0]})', marker='o')
    plt.plot(x, attack_block['predicted_attack_value'], 
             label=f'Predicted ({attack_block["predicted_attack_type"].iloc[0]})', marker='x')
    plt.title('Attack Value Comparison')
    plt.xlabel('Index')
    plt.ylabel('Attack Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
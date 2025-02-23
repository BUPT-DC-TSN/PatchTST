import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from collections import defaultdict
import random


def analyze_attack_data(pd_path):
    # 读取数据
    df = pd.read_csv(pd_path)

    len_df = len(df)
    
    # df = df[int(-0.1*len_df):]

    # 筛选攻击发生数据
    attacked_df = df[df['is_attack'] == 1].copy()
    attacked_df['correct'] = attacked_df['predicted_attack_type'] == attacked_df['attack_type']

    # ========== 基础统计 ==========
    # 攻击类型准确率
    type_accuracy = attacked_df.groupby('attack_type')['correct'].mean().reset_index()
    print("攻击类型准确率：")
    print(type_accuracy.to_string(index=False))

    # 自定义区间分析（含3M以下）
    bins = [0, 3000000, 4000000, np.inf]
    labels = ['≤3M', '3M-4M', '>4M']
    attacked_df['value_range'] = pd.cut(attacked_df['attack_value'], bins=bins, labels=labels, include_lowest=True)
    range_accuracy = attacked_df.groupby('value_range')['correct'].agg(['mean', 'count']).reset_index()
    print("\n攻击值区间准确率：")
    print(range_accuracy.to_string(index=False))

    # ========== 误差分析 ==========
    def calculate_metrics(group):
        metrics = {
            'MAE': mean_absolute_error(group['attack_value'], group['predicted_attack_value']),
            'RMSE': np.sqrt(mean_squared_error(group['attack_value'], group['predicted_attack_value']))
        }
        low_mask = group['attack_value'] <= 3000000
        if low_mask.any():
            metrics.update({
                'MAE_≤3M': mean_absolute_error(group[low_mask]['attack_value'], group[low_mask]['predicted_attack_value']),
                'RMSE_≤3M': np.sqrt(mean_squared_error(group[low_mask]['attack_value'], group[low_mask]['predicted_attack_value']))
            })
        return pd.Series(metrics)

    error_metrics = attacked_df.groupby('attack_type').apply(calculate_metrics).reset_index()
    print("\n数值误差指标：")
    print(error_metrics.to_string(index=False))

    # ========== 连续攻击分析 ==========
    is_attack = df['is_attack'] == 1
    changes = is_attack.ne(is_attack.shift()).cumsum()
    attack_blocks = [group for _, group in df.groupby(changes) if group['is_attack'].all()]

    # ========== 连续攻击分析 ==========
    if not attack_blocks:
        print("\n未找到连续攻击段落")
    else:
        # ===== 初始化统计容器 =====
        step_counts = defaultdict(list)
        block_accuracy = defaultdict(lambda: {'correct':0, 'total':0})
        post_first_correct = defaultdict(lambda: {'correct':0, 'total':0})
        split_accuracy = defaultdict(lambda: {
            'low': {'correct':0, 'total':0},
            'mid': {'correct':0, 'total':0},
            'high': {'correct':0, 'total':0}
        })

        # ===== 遍历所有攻击段落 =====
        for block in attack_blocks:
            attack_type = block['attack_type'].iloc[0]
            same_mask = block['predicted_attack_type'] == block['attack_type']
            
            # 基础统计
            first_match = next((i for i, v in enumerate(same_mask) if v), None)
            is_correct_block = sum(same_mask) >= len(block)//2

            # 需求1：首次匹配步数
            if first_match is not None:
                step_counts[attack_type].append(first_match)

            # 需求2：段落级准确率
            block_accuracy[attack_type]['total'] += 1
            block_accuracy[attack_type]['correct'] += int(is_correct_block)

            # 仅当段落正确时进行后续统计
            if is_correct_block and (first_match is not None):
                # 截取首次正确后的数据
                same_mask_post = same_mask[first_match:]
                post_steps = len(same_mask_post)
                
                # 需求3：首次正确后准确率
                post_first_correct[attack_type]['correct'] += sum(same_mask_post)
                post_first_correct[attack_type]['total'] += post_steps

                # 需求4：阈值分组统计（关键修改部分）
                if attack_type in [1, 2]:
                    # 使用整个段落的最终值进行分组
                    last_value = block['attack_value'].iloc[-1]
                    key = 'high' if last_value > 4000000 else 'mid' if last_value > 3000000 else 'low'
                elif attack_type == 3:
                    # 使用整个段落的均值×2进行分组
                    avg_value = block['attack_value'].mean() * 2
                    key = 'high' if avg_value > 4000000 else 'mid' if avg_value > 3000000 else 'low'
                else:
                    continue
                
                # 仅统计截取部分的正确率
                split_accuracy[attack_type][key]['correct'] += sum(same_mask_post)
                split_accuracy[attack_type][key]['total'] += post_steps

        # ===== 结果输出 =====
        # ...（保持其他输出不变）
        print("\n阈值分组识别准确率（首次正确后）：")
        cond_desc = {
            1: {'low': "≤3M（最终值）", 'mid': "3M-4M（最终值）", 'high': ">4M（最终值）"},
            2: {'low': "≤3M（最终值）", 'mid': "3M-4M（最终值）", 'high': ">4M（最终值）"},
            3: {'low': "≤3M（平均×2）", 'mid': "3M-4M（平均×2）", 'high': ">4M（平均×2）"}
        }
        for atype in sorted(split_accuracy.keys()):
            for range_key in ['low', 'mid', 'high']:
                info = split_accuracy[atype][range_key]
                if info['total'] > 0:
                    acc = info['correct'] / info['total']
                    print(f"类型 {atype}（{cond_desc[atype][range_key]}）: {acc:.2%} ({info['correct']}/{info['total']}步)")

        # ===== 结果输出 =====
        print("\n各攻击类型首次匹配步数（平均值）：")
        for atype in sorted(step_counts):
            avg = np.mean(step_counts[atype])
            print(f"类型 {atype}: {avg:.1f} 步")

        print("\n段落级识别准确率：")
        for atype in sorted(block_accuracy):
            total = block_accuracy[atype]['total']
            acc = block_accuracy[atype]['correct'] / total if total > 0 else 0
            print(f"类型 {atype}: {acc:.2%} ({block_accuracy[atype]['correct']}/{total})")

        print("\n首次正确后识别准确率：")
        for atype in sorted(post_first_correct):
            total = post_first_correct[atype]['total']
            acc = post_first_correct[atype]['correct'] / total if total > 0 else 0
            print(f"类型 {atype}: {acc:.2%} ({post_first_correct[atype]['correct']}/{total})")

        print("\n阈值分组识别准确率：")
        cond_desc = {
            1: {'low': "≤3M（最终值）", 'mid': "3M-4M（最终值）", 'high': ">4M（最终值）"},
            2: {'low': "≤3M（最终值）", 'mid': "3M-4M（最终值）", 'high': ">4M（最终值）"},
            3: {'low': "≤3M（平均×2）", 'mid': "3M-4M（平均×2）", 'high': ">4M（平均×2）"}
        }
        for atype in sorted(split_accuracy.keys()):
            for range_key in ['low', 'mid', 'high']:
                info = split_accuracy[atype][range_key]
                if info['total'] > 0:
                    acc = info['correct'] / info['total']
                    print(f"类型 {atype}（{cond_desc[atype][range_key]}）: {acc:.2%} ({info['correct']}/{info['total']}步)")

    # ========== 可视化 ==========
    attack_type_blocks = {1: [], 2: [], 3: []}
    for block in attack_blocks:
        atype = block['attack_type'].iloc[0]
        if atype in attack_type_blocks:
            attack_type_blocks[atype].append(block)

    plt.figure(figsize=(18, 5))
    for idx, atype in enumerate([1, 2, 3], 1):
        plt.subplot(1, 3, idx)
        blocks = attack_type_blocks.get(atype, [])
        if not blocks:
            continue
        
        block = random.choice(blocks)
        x = block.index
        plt.plot(x, block['attack_value'], label='真实值', marker='o', linewidth=2)
        plt.plot(x, block['predicted_attack_value'], label='预测值', marker='x', linestyle='--')
        
        # 添加阈值线
        plt.axhline(3000000, color='r', linestyle='--', alpha=0.5, label='3M阈值')
        plt.axhline(4000000, color='purple', linestyle='-.', alpha=0.5, label='4M阈值')
        
        plt.title(f'攻击类型 {atype} 实例', fontsize=12)
        plt.xlabel('时间步', fontsize=10)
        plt.ylabel('攻击值', fontsize=10)
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    analyze_attack_data('output_with_predictions22.csv')

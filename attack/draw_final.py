import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 读取数据
df = pd.read_csv('full_predictions_with_value.csv')

# 筛选出攻击发生时的数据
attacked_df = df[df['is_attack'] == 1].copy()

# 计算分类准确率
attacked_df['correct'] = attacked_df['predicted_attack_type'] == attacked_df['attack_type']
accuracy = attacked_df.groupby('attack_type')['correct'].mean().reset_index()
print("攻击类型准确率：")
print(accuracy.to_string(index=False))

# 计算数值误差
error_metrics = attacked_df.groupby('attack_type').apply(
    lambda x: pd.Series({
        'MAE': mean_absolute_error(x['attack_value'], x['predict_attack_value']),
        'RMSE': np.sqrt(mean_squared_error(x['attack_value'], x['predict_attack_value']))
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
    # 取第一个连续攻击段落
    attack_block = attack_blocks[25]
  
    # 绘制图表
    plt.figure(figsize=(12, 6))
  
    x = attack_block.index
    x_label = 'Index'
  
    plt.plot(x, attack_block['attack_value'], 
             label=f'True ({attack_block["attack_type"].iloc[0]})', marker='o')
    plt.plot(x, attack_block['predict_attack_value'], 
             label=f'Predicted ({attack_block["predicted_attack_type"].iloc[0]})', marker='x')
  
    plt.title('Attack Value Comparison')
    plt.xlabel(x_label)
    plt.ylabel('Attack Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
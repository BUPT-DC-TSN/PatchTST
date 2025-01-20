import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 读取CSV文件
df = pd.read_csv('/mnt/e/timer/PatchTST/dataset/0117/train_data.csv')

OT = df.iloc[:, -1]

# 计算每一列与OT的相关系数
correlations = df.corrwith(OT)

# 打印相关系数
print("每一列与OT的相关系数：")
print(correlations)

print(f"mean_ot: {df['OT'].mean()}, std_ot: {df['OT'].std()}")

# 归一化处理
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# 绘制每一列与OT的折线图
for column in df_normalized.columns[:-1]:  # 忽略最后一列OT
    plt.figure(figsize=(10, 6))
    plt.plot(df_normalized.index, df_normalized[column], label=column)
    plt.plot(df_normalized.index, df_normalized.iloc[:, -1], label='OT', linewidth=2, color='black')
    
    plt.xlabel('Index')
    plt.ylabel('Normalized Value')
    plt.title(f'{column} vs OT')
    plt.legend()
    plt.savefig(f'{column}_vs_OT.png')
    plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('timestamp0118_with_pd_1.csv')

# 计算OT列的Z-score
df['OT_zscore'] = (df['OT'] - df['OT'].mean()) / df['OT'].std()

# 过滤掉Z-score绝对值大于3的异常值
df_filtered = df[np.abs(df['OT_zscore']) <= 3]

df_filtered.drop(columns=['OT_zscore'], inplace=False)

# 保存过滤后的数据到新的CSV文件
df_filtered.to_csv('filtered_timestamp0118_with_pd_1.csv', index=False)
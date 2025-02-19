import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 读取数据并解析日期
df = pd.read_csv('timestamp0118_with_pd_1.csv', parse_dates=['date'])

# 计算相关系数
numeric_cols = ['master_offset', 'freq', 'path_delay', 'path_delay_1', 'OT', 'diff_offset']
numeric_cols = ['path_delay', 'path_delay_1', 'OT', 'diff_offset']
correlations = df[numeric_cols].corrwith(df['OT'])
print("各列与OT的相关系数：\n", correlations.to_string(float_format="%.4f"))

# 数据归一化
scaler = StandardScaler()
normalized_data = scaler.fit_transform(df[numeric_cols])
df_normalized = pd.DataFrame(normalized_data, columns=numeric_cols)
df_normalized['date'] = df['date']  # 保留原始日期用于绘图

# 绘制归一化折线图
plt.figure(figsize=(15, 8))
for col in numeric_cols:
    if col != 'OT':
        plt.plot(df_normalized['date'], df_normalized[col], label=col, alpha=0.7)
plt.plot(df_normalized['date'], df_normalized['OT'], 
         label='OT (基准)', linewidth=2, color='black', linestyle='--')

plt.title('归一化特征与OT随时间变化趋势', fontsize=14)
plt.xlabel('日期', fontsize=12)
plt.ylabel('标准化值', fontsize=12)
plt.xticks(rotation=35)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
import pandas as pd

# 读取CSV文件
df = pd.read_csv('/mnt/e/timer/offset/timestamp0118.csv')

# 创建新列path_delay_1，值为前一行path_delay的值
df['path_delay_1'] = df['path_delay'].shift(1)
df['diff_offset'] = df['master_offset'] - df['master_offset'].shift(1)

# 删除首行（无前序数据的行）
df = df.iloc[1:]

# 保存处理后的数据（可修改输出文件名）
df.to_csv('timestamp0118_with_pd_1.csv', index=False)

print("处理完成，结果已保存到 timestamp0118_with_pd_1.csv")
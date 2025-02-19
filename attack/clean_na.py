import pandas as pd

# 读取数据时处理缺失值
def clean_data(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)
  
    # 显示初始数据信息
    print("="*40)
    print("数据清洗前信息:")
    print(f"总行数: {len(df)}")
    print("缺失值统计:")
    print(df.isnull().sum())
  
    # 处理缺失值（删除包含缺失值的行）
    cleaned_df = df.dropna()
  
    # 显示清洗后信息
    print("\n" + "="*40)
    print("数据清洗后信息:")
    print(f"剩余行数: {len(cleaned_df)}")
    print("缺失值统计:")
    print(cleaned_df.isnull().sum())
    print("="*40)
  
    return cleaned_df

# 使用示例
file_path = "/mnt/e/timer/attach/attack.csv"  # 替换为实际路径
cleaned_df = clean_data(file_path)

cleaned_df.to_csv("cleaned_data.csv", index=False)

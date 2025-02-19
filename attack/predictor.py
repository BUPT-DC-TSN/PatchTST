import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 使用相同的随机种子保证可重复性
def set_seed(seed=955):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()
# --------------------------------------------------
# 数据预处理（仅处理攻击样本）
# --------------------------------------------------
# 加载原始数据
full_df = pd.read_csv("/mnt/e/timer/PatchTST/attackcleaned_data.csv")

def create_temporal_features(df, window_size=5):
    """创建时序统计特征"""
    features = []
    for i in range(window_size, len(df)):
        # 原始特征
        current = df.iloc[i]
        # 滑动窗口统计
        window = df.iloc[i-window_size:i]
        stats = [
            window['master_offset'].mean(),    # 均值
            # window['PTPts'].std(),
            window['path_delay'].std(),
            window['s2_freq'].std(),           # 标准差
            window['adjusted_Offset'].std()
        ]
      
        # 合并特征
        combined = np.concatenate([current.values, stats])
        features.append(combined)
  

    return pd.DataFrame(features, columns=list(df.columns) + [
        'rolling_mean', 'rolling_std1', 'rolling_std2', 'rolling_std3',
    ])


# 生成增强特征（在完整数据集上）
window_size=25
enhanced_features = create_temporal_features(full_df.drop(['is_attack', 'attack_value', 'data'], axis=1), window_size=window_size)

# 合并标签并筛选攻击样本
attack_df = enhanced_features[full_df['is_attack'].values[window_size:] == 1].copy()
attack_df['attack_value'] = full_df.loc[full_df['is_attack'] == 1, 'attack_value'].values

# 清除可能的无效数据
attack_df = attack_df.dropna()
print(f"有效攻击样本数: {len(attack_df)}")

# --------------------------------------------------
features = attack_df.drop('attack_value', axis=1)
target = attack_df['attack_value'].values.astype(np.float32)

# 标准化处理（特征和目标值都需要）
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

X_scaled = feature_scaler.fit_transform(features)
y_scaled = target_scaler.fit_transform(target.reshape(-1, 1)).ravel()

# 按时间顺序划分数据集
test_size = 0.2
split_idx = int(len(X_scaled) * (1 - test_size))

X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, 
    test_size=0.2, 
    random_state=42
)

# 创建数据加载器
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)
val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.float32)
)
test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.float32)
)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)


class DualBranchPredictor(nn.Module):
    def __init__(self, raw_dim, stat_dim):
        super().__init__()
      
        # 原始特征分支
        self.raw_dim = raw_dim
        self.stat_dim = stat_dim
        self.raw_branch = nn.Sequential(
            nn.Linear(raw_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )
      
        # 统计特征分支（使用时序处理）
        self.stat_branch = nn.Sequential(
            nn.Linear(stat_dim, 32),
            nn.LSTM(input_size=32, hidden_size=32, batch_first=True),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Dropout(0.2)
        )
      
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(64, 32),  # 32+32=64
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # 处理原始特征
        raw = x[:, :self.raw_dim]
        stat = x[:, -self.stat_dim:]
        raw_out = self.raw_branch(raw)
      
        # 处理统计特征（添加时间维度）
        stat = stat.unsqueeze(1)  # [batch, 1, stat_dim]
        stat_out, _ = self.stat_branch[1](self.stat_branch[0](stat))
        # stat_out = self.stat_branch(stat)[:, -1, :]
        stat_out = self.stat_branch[2:](stat_out[:, -1, :])
      
        # 特征融合
        combined = torch.cat([raw_out, stat_out], dim=1)
        return self.fusion(combined).squeeze()


# class AttackValuePredictor(nn.Module):
#     def __init__(self, input_size):
#         super().__init__()
#         # 移除硬编码的time_steps参数
#         self.feature_net = nn.Sequential(
#             nn.BatchNorm1d(input_size),
#             nn.Dropout(0.3),
#             nn.Linear(input_size, 128),
#             nn.GELU()
#         )
      
#         # 自适应时序处理
#         # self.temporal_layer = nn.LSTM(
#         #     input_size=128,
#         #     hidden_size=128,
#         #     num_layers=1,  # 减少层数
#         #     batch_first=True
#         # )
        
        
        
#         self.regressor = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.GELU(),
#             nn.Dropout(0.2),
#             nn.Linear(64, 32),
#             nn.GELU(),
#             nn.Dropout(0.2),
#             nn.Linear(32, 1)
#         )

#     def forward(self, x):
#         # 特征增强
#         x = self.feature_net(x)  # [batch, 128]
      
#         # 添加时间维度（模拟时序）
#         x = x.unsqueeze(1)  # [batch, 1, 128]
      
#         # 时序处理
#         # temporal_out, _ = self.temporal_layer(x)
#         # temporal_out, _ = self.temporal_attn(temporal_out, temporal_out, temporal_out)
#         # temporal_out = temporal_out[:, -1, :]  # 取最后时间步
      
#         # return self.regressor(temporal_out).squeeze()
#         return self.regressor(x).squeeze()

# 初始化时不再需要time_steps参数
# model = AttackValuePredictor(input_size=X_train.shape[1])
model = DualBranchPredictor(raw_dim=X_train.shape[1]-4, stat_dim=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# --------------------------------------------------
# 训练配置
# --------------------------------------------------
criterion = nn.HuberLoss(delta=1.0)  # 结合MAE和MSE优点
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

best_val_loss = float('inf')
early_stop_patience = 20
patience_counter = 0

# --------------------------------------------------
# 训练循环
# --------------------------------------------------
for epoch in range(200):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
      
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step()
      
        train_loss += loss.item() * inputs.size(0)
  
    # 验证阶段
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item() * inputs.size(0)
  
    # 学习率调度
    train_loss = train_loss / len(train_loader.dataset)
    val_loss = val_loss / len(val_loader.dataset)
    scheduler.step(val_loss)
  
    # 早停机制
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_regressor.pth")
    else:
        patience_counter += 1
  
    print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
  
    if patience_counter >= early_stop_patience:
        print("Early stopping triggered")
        break

# --------------------------------------------------
# 模型评估
# --------------------------------------------------
model.load_state_dict(torch.load("best_regressor.pth"))
model.eval()

def inverse_scale(y, scaler):
    return scaler.inverse_transform(y.reshape(-1, 1)).ravel()

with torch.no_grad():
    # 测试集预测
    test_preds, test_labels = [], []
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs).cpu().numpy()
        test_preds.extend(outputs)
        test_labels.extend(labels.numpy())
  
    # 反标准化
    test_preds = inverse_scale(np.array(test_preds), target_scaler)
    test_labels = inverse_scale(np.array(test_labels), target_scaler)
  
    # 计算指标
    mae = mean_absolute_error(test_labels, test_preds)
    rmse = np.sqrt(mean_squared_error(test_labels, test_preds))
    r2 = r2_score(test_labels, test_preds)
  
    print(f"\nFinal Test Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")

    model.eval()
    # 在模型评估部分之后添加
    plt.figure(figsize=(12, 6))
    plt.plot(test_labels, label='True Values', color='blue', alpha=0.7, linewidth=2)
    plt.plot(test_preds, label='Predictions', color='red', linestyle='--', alpha=0.9)
    plt.title('Attack Value Prediction Comparison', fontsize=14)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Attack Value', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # 保存并显示图像
    plt.savefig('prediction_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# --------------------------------------------------
# 示例预测函数
# --------------------------------------------------
def predict_attack_value(sample_tensor):
    model.eval()
    with torch.no_grad():
        sample_tensor = sample_tensor.to(device)
        prediction = model(sample_tensor.unsqueeze(0))
        return inverse_scale(prediction.cpu().numpy(), target_scaler)[0]
    



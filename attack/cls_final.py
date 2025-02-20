import datetime
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# 设置随机种子
def set_seed(seed=955):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

# --------------------------------------------------
# 数据预处理（关键修改）
# --------------------------------------------------
full_df = pd.read_csv("/mnt/e/timer/attack/fixed_data.csv")

def create_temporal_features(df, window_size=5):
    """创建时序特征并确保维度可被窗口大小整除"""
    features = []
    label_indices = []
  
    for i in range(window_size, len(df)):
        # 当前时刻特征
        current = df.iloc[i].values
      
        # 窗口统计特征（扩展更多统计量）
        window = df.iloc[i-window_size:i]
        stats = []
        for col in ['master_offset', 'path_delay', 's2_freq', 'adjusted_Offset']:
            stats.extend([
                window[col].mean(),
                window[col].std(),
                window[col].max() - window[col].min(),
                window[col].quantile(0.25)
            ])
      
        # 合并特征（当前时刻+窗口统计）
        combined = np.concatenate([current, stats])
        features.append(combined)
        label_indices.append(i)
  
    return pd.DataFrame(features), label_indices

# 生成特征 20比较好
window_size = 20
features_df = full_df.drop(['is_attack', 'attack_value', 'data', 'attack_type'], axis=1)
enhanced_features, label_indices = create_temporal_features(features_df, window_size)

# 维度验证
total_features = enhanced_features.shape[1]
expected_temporal_features = 4 * 4  # 4个特征列，每个计算4个统计量
assert ((total_features - features_df.shape[1]) == expected_temporal_features), (f"特征生成错误，期望增加{expected_temporal_features}个特征，实际增加{total_features - features_df.shape[1]}")

# 对齐标签
labels = full_df.iloc[label_indices]['attack_type'].values.astype(np.int64)
valid_indices = ~enhanced_features.isna().any(axis=1)
X = enhanced_features[valid_indices].values
y = labels[valid_indices]

# --------------------------------------------------
# 数据标准化与划分
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分层划分
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, 
    test_size=0.1, 
    stratify=y,
    random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=42
)

# 创建数据集
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

batch_size = 512
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# --------------------------------------------------
# 模型架构（维度自适应）
# --------------------------------------------------
class TemporalAttackDetector(nn.Module):
    def __init__(self, input_size, time_steps=1):
        super().__init__()
        self.time_steps = time_steps
        self.input_size = input_size
      
        # 时序特征处理分支
        self.temporal_net = nn.LSTM(
            input_size=input_size,  # 原始特征维度
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.temporal_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4)
      
        # 统计特征处理分支
        self.stats_net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.GELU(),
            nn.LayerNorm(32)
        )
        
        
        # 联合分类器
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 4)
        )
      
        # 辅助网络
        self.bn = nn.BatchNorm1d(input_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x = self.bn(x)
        # x = self.dropout(x)
      
        # 分割特征
        primary = x[:,]  # [batch, input_size-4]
        stats = x[:,]    # [batch, 4]
      
        # 时序特征处理
        batch_size = x.size(0)
        primary = primary.view(batch_size, self.time_steps, -1)  # 重组为时序数据
        temporal_out, _ = self.temporal_net(primary)  # [batch, seq_len, 128]
      
        # 注意力机制
        temporal_out = temporal_out.permute(1, 0, 2)  # [seq_len, batch, features]
        attn_out, _ = self.temporal_attn(temporal_out, temporal_out, temporal_out)
        temporal_feat = attn_out[-1]  # 取最后一个时间步 [batch, 128]
      
        # 统计特征处理
        stats_feat = self.stats_net(stats)  # [batch, 32]
      
        # 特征融合
        combined = torch.cat([temporal_feat, stats_feat], dim=1)
        return self.classifier(combined).squeeze()


# 初始化模型
input_size = X_train.shape[1]
model = TemporalAttackDetector(input_size)

# --------------------------------------------------
# 训练配置
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
#                                                       mode='max', 
#                                                       factor=0.7,
#                                                       patience=5,
#                                                       verbose=True)

# # --------------------------------------------------
# # 训练循环
# # --------------------------------------------------
# best_val_acc = 0.0
# patience = 15
# early_stop_counter = 0

# for epoch in range(100):
#     # 训练阶段
#     model.train()
#     train_loss = 0.0
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
      
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
  
#     # 验证阶段
#     model.eval()
#     val_loss = 0.0
#     correct = 0
#     with torch.no_grad():
#         for inputs, labels in val_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             val_loss += criterion(outputs, labels).item()
#             preds = torch.argmax(outputs, dim=1)
#             correct += (preds == labels).sum().item()
  
#     # 打印统计信息
#     train_loss /= len(train_loader)
#     val_loss /= len(val_loader)
#     val_acc = correct / len(val_loader.dataset)
#     scheduler.step(val_acc)
  
#     print(f"Epoch {epoch+1:02d} | "
#           f"Train Loss: {train_loss:.4f} | "
#           f"Val Loss: {val_loss:.4f} | "
#           f"Val Acc: {val_acc:.4f}")
  
#     # 早停机制
#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         torch.save(model.state_dict(), "cls_model.pth")
#         early_stop_counter = 0
#     else:
#         early_stop_counter += 1
#         if early_stop_counter >= patience:
#             print("Early stopping triggered")
#             break

# # --------------------------------------------------
# # 最终评估
# # --------------------------------------------------
# model.load_state_dict(torch.load("cls_model.pth"))
# model.eval()

# joblib.dump(scaler, 'cls_scaler.pkl')


# all_preds = []
# all_labels = []
# with torch.no_grad():
#     for inputs, labels in test_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = model(inputs)
#         preds = torch.argmax(outputs, dim=1)
#         all_preds.extend(preds.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())

# print("\nClassification Report:")
# print(classification_report(all_labels, all_preds))
# print(f"Test Accuracy: {np.mean(np.array(all_preds) == np.array(all_labels)):.4f}")


# 全量预测与保存（新增部分）
# --------------------------------------------------
# 创建完整数据集
full_dataset = TensorDataset(torch.FloatTensor(X_scaled))
full_loader = DataLoader(full_dataset, batch_size=batch_size)

# 全量预测
model.load_state_dict(torch.load("cls_model.pth"))
model.eval()
model.to(device)

all_preds = []
with torch.no_grad():
    for inputs in full_loader:
        inputs = inputs[0].to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())

# 创建结果DataFrame（处理原始索引）
result_df = full_df.copy()
result_df['predicted_attack_type'] = np.nan
result_df['predicted_attack_type'][window_size:] = all_preds

result_df = result_df.dropna()

# 保存结果
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_filename = f'full_cls_predictions.csv'
result_df.to_csv(result_filename, index=False)

print(f"\n完整预测结果已保存至 {result_filename}")
print(f"有效预测样本数：{len(all_preds)}/{len(full_df)}")
print(f"缺失预测的行：{result_df['predicted_attack_type'].isna().sum()} 行（窗口效应导致）")
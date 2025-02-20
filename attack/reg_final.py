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
import joblib


# 设置随机种子保证可重复性
def set_seed(seed=955):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

# --------------------------------------------------
# 数据预处理
# --------------------------------------------------
def create_temporal_features(df, window_size=25):
    """创建时序统计特征"""
    features = []
    label_indices = []
  
    for i in range(window_size, len(df)):
        current = df.iloc[i].values
        window = df.iloc[i-window_size:i]
      
    #     stats = [
    #         window['master_offset'].mean(),
    #         window['path_delay'].std(),
    #         window['s2_freq'].std(),
    #         window['adjusted_Offset'].std()
    #     ]
      
    #     features.append(np.concatenate([current, stats]))
    #     label_indices.append(i)
  
    # return pd.DataFrame(features), label_indices
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

# 加载数据
full_df = pd.read_csv("/mnt/e/timer/attack/full_cls_predictions.csv")
features_df = full_df.drop(['is_attack', 'attack_value', 'data', 'predicted_attack_type'], axis=1)
window_size = 20

# 生成时序特征
enhanced_features, label_indices = create_temporal_features(features_df, window_size)

# 对齐标签和攻击类型
attack_mask = full_df['is_attack'].values[window_size:] == 1
attack_df = enhanced_features[attack_mask].copy()
attack_df['attack_value'] = full_df.loc[full_df['is_attack'] == 1, 'attack_value'].values
attack_df['attack_type'] = full_df.loc[full_df['is_attack'] == 1, 'attack_type'].values
attack_df['predicted_attack_type'] = full_df.loc[full_df['is_attack'] == 1, 'predicted_attack_type'].values
attack_df = attack_df.dropna()

# 数据标准化
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

X = feature_scaler.fit_transform(attack_df.drop(['attack_value', 'attack_type', 'predicted_attack_type'], axis=1))
y = target_scaler.fit_transform(attack_df['attack_value'].values.reshape(-1, 1)).ravel()
types = attack_df['attack_type'].values.astype(np.int64)

pred_types = attack_df['predicted_attack_type'].values.astype(np.int64)

# 数据集划分
X_train, X_test, y_train, y_test, types_train, types_test = train_test_split(
    X, y, types, test_size=0.1, random_state=42, stratify=types
)
X_train, X_val, y_train, y_val, types_train, types_val = train_test_split(
    X_train, y_train, types_train, test_size=0.25, random_state=42, stratify=types_train
)

class TypeSpecificModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
      
        # 共享特征提取层
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1)
        )
      
        # 类型特定分支
        self.branches = nn.ModuleDict({
            '1': self._build_linear_branch(),
            '2': self._build_linear_branch(),
            '3': self._build_random_branch()
        })
  
    def _build_linear_branch(self):
        return nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1)
        )
  
    def _build_random_branch(self):
        return nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
  
    def forward(self, x, attack_type):
        shared = self.shared_layer(x)
      
        outputs = torch.zeros(x.size(0), 1, device=x.device)
        for type_id in ['1', '2', '3']:
            mask = (attack_type == int(type_id))
            if mask.any():
                outputs[mask] = self.branches[type_id](shared[mask])
      
        return outputs.squeeze()

# --------------------------------------------------
# 训练配置
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(X_train.shape)
model = TypeSpecificModel(input_dim=X_train.shape[1]).to(device)

# 创建数据加载器
def create_loaders(X, y, types, batch_size=512, shuffle=True):
    dataset = TensorDataset(
        torch.FloatTensor(X),
        torch.LongTensor(types),
        torch.FloatTensor(y)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_loader = create_loaders(X_train, y_train, types_train)
val_loader = create_loaders(X_val, y_val, types_val, shuffle=False)
test_loader = create_loaders(X_test, y_test, types_test, shuffle=False)

criterion = nn.HuberLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

# --------------------------------------------------
# 训练循环
# --------------------------------------------------
best_val_loss = float('inf')
early_stop_patience = 15
patience_counter = 0

# for epoch in range(200):
#     # 训练阶段
#     model.train()
#     train_loss = 0.0
#     for inputs, types, labels in train_loader:
#         inputs, types, labels = inputs.to(device), types.to(device), labels.to(device)
      
#         optimizer.zero_grad()
#         outputs = model(inputs, types)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()
      
#         train_loss += loss.item() * inputs.size(0)
  
#     # 验证阶段
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for inputs, types, labels in val_loader:
#             inputs, types, labels = inputs.to(device), types.to(device), labels.to(device)
#             outputs = model(inputs, types)
#             val_loss += criterion(outputs, labels).item() * inputs.size(0)
  
#     # 计算平均损失
#     train_loss /= len(train_loader.dataset)
#     val_loss /= len(val_loader.dataset)
#     scheduler.step(val_loss)
  
#     # 早停机制
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         patience_counter = 0
#         torch.save(model.state_dict(), "reg_model.pth")
#     else:
#         patience_counter += 1
  
#     print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
  
#     if patience_counter >= early_stop_patience:
#         print("Early stopping triggered")
#         break

# --------------------------------------------------
# 模型评估
# --------------------------------------------------
model.load_state_dict(torch.load("reg_model.pth"))
model.eval()

def inverse_scale(y, scaler):
    return scaler.inverse_transform(y.reshape(-1, 1)).ravel()

test_preds, test_labels = [], []
with torch.no_grad():
    for inputs, types, labels in train_loader:
        inputs, types = inputs.to(device), types.to(device)
        outputs = model(inputs, types).cpu().numpy()
        test_preds.extend(outputs)
        test_labels.extend(labels.numpy())

test_preds = inverse_scale(np.array(test_preds), target_scaler)
test_labels = inverse_scale(np.array(test_labels), target_scaler)


joblib.dump(feature_scaler, 'reg_feature_scaler.pkl')
joblib.dump(target_scaler, 'reg_target_scaler.pkl')


# 计算指标
mae = mean_absolute_error(test_labels, test_preds)
rmse = np.sqrt(mean_squared_error(test_labels, test_preds))
r2 = r2_score(test_labels, test_preds)

print(f"\nFinal Test Performance:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

# # 可视化结果
# plt.figure(figsize=(12, 6))
# plt.plot(test_labels, label='True Values', alpha=0.8)
# plt.plot(test_preds, label='Predictions', alpha=0.8)
# plt.title("Attack Value Prediction Results")
# plt.xlabel("Sample Index")
# plt.ylabel("Attack Value")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('prediction_results.png', dpi=300)
# plt.show()

full_dataset = TensorDataset(
    torch.FloatTensor(X),  # 攻击样本特征
    torch.LongTensor(pred_types)  # 对应的攻击类型
)
full_loader = DataLoader(full_dataset, batch_size=128, shuffle=False)

model.to(device)

attack_preds = []
with torch.no_grad():
    for inputs, types_batch in full_loader:
        inputs, types_batch = inputs.to(device), types_batch.to(device)
        preds = model(inputs, types_batch)  # 同时传入特征和攻击类型
        attack_preds.extend(preds.cpu().numpy().flatten())  # 修正变量名错误

# 逆标准化预测结果
attack_preds = target_scaler.inverse_transform(np.array(attack_preds).reshape(-1, 1)).flatten()

# 将预测结果对齐到原始数据集
result_df = full_df.copy()

# 初始化预测列（使用float类型以兼容NaN）
result_df['predict_attack_value'] = np.nan

# 获取攻击点在原始数据中的真实位置
# label_indices是create_temporal_features生成的窗口结束位置索引
# attack_mask是full_df[window_size:]中标记为攻击的点
attack_indices = np.array(label_indices)[attack_mask]  # 转换为numpy数组便于索引操作

# 确保索引数量与预测值一致
assert len(attack_indices) == len(attack_preds), f"索引数量({len(attack_indices)})与预测值数量({len(attack_preds)})不匹配"

# 将预测值填入对应位置
result_df.loc[attack_indices, 'predict_attack_value'] = attack_preds

# 保存结果
result_filename = 'full_predictions_with_value.csv'
result_df.to_csv(result_filename, index=False)
print(f"预测结果已保存至 {result_filename}")



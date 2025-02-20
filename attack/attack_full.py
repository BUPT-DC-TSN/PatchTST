import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
import joblib
from datetime import datetime

# 设置全局随机种子
def set_seed(seed=955):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
set_seed()

# ----------------------------
# 分类模型部分（攻击类型预测）
# ----------------------------
def create_temporal_features(df, window_size=20):
    """创建时序特征"""
    features = []
    label_indices = []
    for i in range(window_size, len(df)):
        current = df.iloc[i].values
        window = df.iloc[i-window_size:i]
        stats = []
        for col in ['master_offset', 'path_delay', 's2_freq', 'adjusted_Offset']:
            stats.extend([
                window[col].mean(),
                window[col].std(),
                window[col].max() - window[col].min(),
                window[col].quantile(0.25)
            ])
        combined = np.concatenate([current, stats])
        features.append(combined)
        label_indices.append(i)
    return pd.DataFrame(features), label_indices

# 加载原始数据
full_df = pd.read_csv("/mnt/e/timer/attack/fixed_data.csv")

# 生成分类特征
features_df = full_df.drop(['is_attack', 'attack_value', 'data', 'attack_type'], axis=1)
window_size = 20
enhanced_features, label_indices = create_temporal_features(features_df, window_size)

# 处理有效索引
valid_indices = ~enhanced_features.isna().any(axis=1)
X_cls = enhanced_features[valid_indices].values
y_cls = full_df.iloc[label_indices]['attack_type'].values.astype(np.int64)[valid_indices]
valid_original_indices = np.array(label_indices)[valid_indices]

# 分类数据标准化
cls_scaler = StandardScaler()
X_cls_scaled = cls_scaler.fit_transform(X_cls)

# 分类模型定义
class TemporalAttackDetector(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.temporal_net = nn.LSTM(input_size, 64, num_layers=2, batch_first=True, bidirectional=True)
        self.temporal_attn = nn.MultiheadAttention(128, 4)
        self.stats_net = nn.Sequential(nn.Linear(input_size, 32), nn.GELU(), nn.LayerNorm(32))
        self.classifier = nn.Sequential(nn.Linear(160, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, 4))
        self.bn = nn.BatchNorm1d(input_size)
        self.dropout = nn.Dropout(0.2)
  
    def forward(self, x):
        x = self.bn(x)
        x = self.dropout(x)
        temporal_out, _ = self.temporal_net(x.unsqueeze(1))
        temporal_out = temporal_out.permute(1, 0, 2)
        attn_out, _ = self.temporal_attn(temporal_out, temporal_out, temporal_out)
        temporal_feat = attn_out[-1]
        stats_feat = self.stats_net(x)
        combined = torch.cat([temporal_feat, stats_feat], dim=1)
        return self.classifier(combined)

# 分类模型训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cls_model = TemporalAttackDetector(X_cls_scaled.shape[1]).to(device)

# 数据划分
X_train, X_temp, y_train, y_temp = train_test_split(
    X_cls_scaled, y_cls, test_size=0.1, stratify=y_cls, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# 创建数据加载器
def cls_loader(X, y, batch_size=512, shuffle=True):
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_loader = cls_loader(X_train, y_train)
val_loader = cls_loader(X_val, y_val, shuffle=False)
test_loader = cls_loader(X_test, y_test, shuffle=False)

# 训练配置
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cls_model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.7, patience=5)

# 训练循环
best_acc = 0.0
early_stop = 15
counter = 0

for epoch in range(100):
    cls_model.train()
    total_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = cls_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
  
    # 验证
    cls_model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = cls_model(inputs)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
  
    val_acc = correct / len(val_loader.dataset)
    scheduler.step(val_acc)
  
    # 早停机制
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(cls_model.state_dict(), "cls_model_1.pth")
        counter = 0
    else:
        counter += 1
  
    print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")
  
    if counter >= early_stop:
        print("Early stopping")
        break

# 全量分类预测
full_cls_loader = DataLoader(TensorDataset(torch.FloatTensor(X_cls_scaled)), batch_size=512)
cls_model.load_state_dict(torch.load("cls_model_1.pth"))
cls_model.eval()

cls_preds = []
with torch.no_grad():
    for batch in full_cls_loader:
        outputs = cls_model(batch[0].to(device))
        cls_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())

# 保存分类结果
result_df = full_df.copy()
result_df['predicted_attack_type'] = np.nan
result_df.iloc[valid_original_indices, -1] = cls_preds
result_df['predicted_attack_type'] = result_df['predicted_attack_type'].fillna(0).astype(int)

# ----------------------------
# 回归模型部分（攻击值预测）
# ----------------------------
def prepare_regression_features(df, window_size=20):
    features = []
    label_indices = []
    for i in range(window_size, len(df)):
        current = df.iloc[i][['master_offset', 'path_delay', 's2_freq', 'adjusted_Offset', 'predicted_attack_type']].values
        window = df.iloc[i-window_size:i]
        stats = []
        for col in ['master_offset', 'path_delay', 's2_freq', 'adjusted_Offset']:
            stats.extend([
                window[col].mean(),
                window[col].std(),
                window[col].max() - window[col].min(),
                window[col].quantile(0.25)
            ])
        combined = np.concatenate([current, stats])
        features.append(combined)
        label_indices.append(i)
    return pd.DataFrame(features), label_indices

# 生成回归特征
reg_features, reg_label_indices = prepare_regression_features(result_df)
valid_reg_indices = ~reg_features.isna().any(axis=1)
X_reg = reg_features[valid_reg_indices].values
valid_reg_original_indices = np.array(reg_label_indices)[valid_reg_indices]

# 回归数据标准化
reg_scaler = StandardScaler()
X_reg_scaled = reg_scaler.fit_transform(X_reg[:, :-1])  # 最后一列为预测类型

# 回归模型定义
class AttackValuePredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.branches = nn.ModuleDict({
            str(i): nn.Sequential(
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            ) for i in range(1, 4)
        })
  
    def forward(self, x, attack_types):
        shared = self.shared(x)
        outputs = torch.zeros(x.size(0), 1, device=x.device)
        for type_id in ['1', '2', '3']:
            mask = (attack_types == int(type_id))
            if mask.any():
                outputs[mask] = self.branches[type_id](shared[mask])
        return outputs.squeeze()

# 回归模型训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reg_model = AttackValuePredictor(X_reg_scaled.shape[1]).to(device)

# 数据准备
y_reg = full_df.loc[valid_reg_original_indices, 'attack_value'].values.astype(np.float32)
types_reg = X_reg[:, -1].astype(int)

# 数据划分
X_train, X_test, y_train, y_test, types_train, types_test = train_test_split(
    X_reg_scaled, y_reg, types_reg, test_size=0.2, random_state=42, stratify=types_reg
)

# 数据加载器
def reg_loader(X, y, types, batch_size=512, shuffle=True):
    dataset = TensorDataset(
        torch.FloatTensor(X),
        torch.LongTensor(types),
        torch.FloatTensor(y)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_loader = reg_loader(X_train, y_train, types_train)
test_loader = reg_loader(X_test, y_test, types_test, shuffle=False)

# 训练配置
criterion = nn.HuberLoss()
optimizer = torch.optim.AdamW(reg_model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

# 训练循环
best_loss = float('inf')
early_stop = 15
counter = 0

for epoch in range(200):
    reg_model.train()
    total_loss = 0.0
    for inputs, types, labels in train_loader:
        inputs, types, labels = inputs.to(device), types.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = reg_model(inputs, types)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(reg_model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
  
    # 验证
    reg_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, types, labels in test_loader:
            inputs, types, labels = inputs.to(device), types.to(device), labels.to(device)
            outputs = reg_model(inputs, types)
            val_loss += criterion(outputs, labels).item()
  
    val_loss /= len(test_loader)
    scheduler.step(val_loss)
  
    # 早停机制
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(reg_model.state_dict(), "reg_model_1.pth")
        counter = 0
    else:
        counter += 1
  
    print(f"Epoch {epoch+1:03d} | Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
  
    if counter >= early_stop:
        print("Early stopping")
        break

# 全量回归预测
reg_loader = DataLoader(TensorDataset(
    torch.FloatTensor(reg_scaler.transform(X_reg[:, :-1])),
    torch.LongTensor(X_reg[:, -1].astype(int))
), batch_size=512)

reg_model.load_state_dict(torch.load("reg_model_1.pth"))
reg_model.eval()

attack_values = []
with torch.no_grad():
    for inputs, types in reg_loader:
        inputs, types = inputs.to(device), types.to(device)
        outputs = reg_model(inputs, types)
        attack_values.extend(outputs.cpu().numpy())

# 反标准化
target_scaler = StandardScaler()
target_scaler.fit(full_df[full_df['is_attack'] == 1]['attack_value'].values.reshape(-1, 1))
attack_values = target_scaler.inverse_transform(np.array(attack_values).reshape(-1, 1)).flatten()

# 保存最终结果
result_df['predicted_attack_value'] = 0.0
result_df.loc[valid_reg_original_indices, 'predicted_attack_value'] = attack_values
result_df['predicted_attack_value'] = result_df.apply(
    lambda x: x['predicted_attack_value'] if x['predicted_attack_type'] != 0 else 0,
    axis=1
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_df.to_csv(f'full_predictions_{timestamp}.csv', index=False)
print(f"最终结果已保存至 full_predictions_{timestamp}.csv")
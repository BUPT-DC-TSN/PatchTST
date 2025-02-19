import pandas as pd
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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
            window['s2_freq'].std(),           # 标准差
            window['path_delay'].std(),
            window['adjusted_Offset'].std(),
            # window['path_delay'].max() - window['path_delay'].min(),  # 极差
            # (window['adjusted_Offset'].diff().abs() > 1000).sum()     # 突变次数
        ]
      
        # 合并特征（保留原始特征）
        combined = np.concatenate([current.values, stats])
        features.append(combined)
  
    return pd.DataFrame(features, columns=list(df.columns) + [
        'rolling_mean', 'rolling_std', 'rolling_range', 'change_count'
    ])

# 加载数据并生成时序特征
df = pd.read_csv("/mnt/e/timer/PatchTST/attackcleaned_data.csv")
window_size=5
enhanced_df = create_temporal_features(df.drop(['is_attack', 'attack_value', 'data'], axis=1), window_size=window_size)

# enhanced_df = create_temporal_features(df.drop(['attack_value', 'PTPts'], axis=1))
# enhanced_df.to_csv("/mnt/e/timer/PatchTST/attackenhanced_data.csv", index=False)

# 合并标签并清除无效数据
enhanced_df['is_attack'] = df['is_attack'].values[window_size:]  # 对齐标签
enhanced_df = enhanced_df.dropna()

# 2. 数据集划分
# --------------------------------------------------
features = enhanced_df.drop(['is_attack'], axis=1)
labels = enhanced_df['is_attack'].values.astype(np.float32)

# 标准化
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


X_train, X_temp, y_train, y_temp = train_test_split(
    scaled_features, labels, 
    test_size=0.3, 
    stratify=labels,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=42
)


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


batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

# --------------------------------------------------
class TemporalAttackDetector(nn.Module):
    def __init__(self, input_size, time_steps=5):
        super().__init__()
        self.time_steps = time_steps
        self.input_size = input_size
      
        # 时序特征处理分支
        self.temporal_net = nn.LSTM(
            input_size=input_size-4,  # 原始特征维度
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.temporal_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4)
      
        # 统计特征处理分支
        self.stats_net = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.LayerNorm(32)
        )
      
        # 联合分类器
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
      
        # 辅助网络
        self.bn = nn.BatchNorm1d(input_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.bn(x)
        x = self.dropout(x)
      
        # 分割特征
        primary = x[:, :-4]  # [batch, input_size-4]
        stats = x[:, -4:]    # [batch, 4]
      
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

# 修改模型初始化部分
input_size = X_train.shape[1]
time_steps = 1  # 与特征生成时的窗口大小一致
model = TemporalAttackDetector(input_size, time_steps=time_steps)

# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 处理类别不平衡
pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / sum(y_train)], 
                         dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --------------------------------------------------
best_val_acc = 0.0
num_epochs = 100
patience=100
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
    
    # 验证阶段
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            
            # 计算准确率
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    # 打印统计信息
    train_loss = train_loss / len(train_loader.dataset)
    val_loss = val_loss / len(val_loader.dataset)
    val_acc = correct / total
    
    print(f"Epoch {epoch+1:3d} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc:.4f}")
    
    # 早停机制和模型保存（修改部分）
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        early_stop_counter = 0  # 重置计数器
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement.")
            break  # 终止训练循环

# --------------------------------------------------
# 加载最佳模型
model.load_state_dict(torch.load("best_model.pth"))

# 测试集评估
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs).squeeze()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nTest Results:")
print(classification_report(all_labels, all_preds))
print(f"Final Accuracy: {np.mean(np.array(all_preds) == np.array(all_labels)):.4f}")


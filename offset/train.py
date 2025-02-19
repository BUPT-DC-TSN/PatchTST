import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import csv
from kalman import show_kalman_filter


class StdLoss(nn.Module):
    def __init__(self):
        super(StdLoss, self).__init__()
    
    def forward(self, outputs, targets, k=0.7):
        mse_loss = nn.MSELoss()(outputs, targets)
        # 计算一个batch内模型的均值
        # mean_t = torch.mean(targets)
        # mean_o = torch.mean(outputs)
        # diff_outputs = torch.abs(outputs - mean_o)
        # diff_targets = torch.abs(targets - mean_t)
        # std_loss = nn.MSELoss()(diff_outputs, diff_targets)
        mae_loss = nn.L1Loss()(outputs, targets)
        
        # return mse_loss + k * std_loss
        # return mse_loss + mae_loss + k*std_loss
        return k*mse_loss + (1-k)*mae_loss

    
    # 不合理，对两部分分别设置损失项进行惩罚，该如何设置？

# 固定随机种子
def set_seed(seed=2025):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def preprocess_data(file_path):
    # 读取数据
    data = pd.read_csv(file_path)
    
    # cols = data.columns
    
    # cols.drop('master_offset', 'freq')
    
    # data = data[cols + ['master_offset'] + ['freq']]

    # 将日期列转换为秒数（假设日期是相对于某个固定时间点的秒数）
    data['date'] = pd.to_datetime(data['date'])
    data['Seconds'] = (data['date'] - data['date'].min()).dt.total_seconds()

    # 添加 Seconds_sin 和 Seconds_cos 特征
    data['Seconds_sin'] = np.sin(2 * np.pi * data['Seconds'] / (24 * 3600))
    data['Seconds_cos'] = np.cos(2 * np.pi * data['Seconds'] / (24 * 3600))

    # data = data.drop(['master_offset', 'freq', 'date'], axis=1)
    # X = data[['master_offset', 'freq', 'Seconds_sin', 'Seconds_cos']].values
    # X = data[['master_offset', 'freq']].values
    # X = data[['path_delay']].values
    # X = data[['path_delay_1', 'Seconds_sin', 'Seconds_cos', 'path_delay', 'master_offset', 'freq']].values
    X = data[['path_delay', 'path_delay_1', 'Seconds_sin', 'Seconds_cos', 'master_offset', 'freq']].values
    # X = data[['path_delay', 'Seconds_sin', 'Seconds_cos', 'master_offset', 'freq']].values
    # X = data[['path_delay', 'master_offset', 'freq']].values
    
    y = data['OT'].values

    # 标准化特征
    scaler = StandardScaler()
    # xscaler = StandardScaler()
    # yscaler = StandardScaler()
    X = scaler.fit_transform(X)
    # y = yscaler.fit_transform(y)
    # data_max = data['OT'].values.max()
    # data_min = data['OT'].values.min()
    data_mean = data['OT'].values.mean()
    data_std = data['OT'].values.std()
    y = (y - data_mean) / data_std
    # y = (y - data_min) / (data_max - data_min)

    # 将数据分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.6, random_state=42
    )

    # 转换为 PyTorch 张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # return X_train, X_test, y_train, y_test, data_max, data_min, X, y
    return X_train, X_test, y_train, y_test, data_mean, data_std, X, y

# 创建自定义 Dataset 和 DataLoader
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class GDNNOld(nn.Module):
    def __init__(self, input_size1, input_size, hidden_size, output_size):
        super(GDNNOld, self).__init__()
        self.input_size1 = input_size1
        self.fc11 = nn.Sequential(
            nn.Linear(input_size1, 4*hidden_size),
            nn.BatchNorm1d(4*hidden_size),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # nn.Tanh(),
            nn.Linear(4*hidden_size, 2*hidden_size),
            nn.BatchNorm1d(2*hidden_size),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # nn.Tanh(),
            nn.Linear(2*hidden_size, hidden_size)
        )
        self.fc12 = nn.Sequential(
            nn.Linear(input_size, 4*hidden_size),
            nn.BatchNorm1d(4*hidden_size),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # nn.Tanh(),
            nn.Linear(4*hidden_size, 2*hidden_size),
            nn.BatchNorm1d(2*hidden_size),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # nn.Tanh(),
            nn.Linear(2*hidden_size, hidden_size)
        )
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.drop = nn.Dropout(p=0.1)
        self.norm = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, 8*hidden_size),
            nn.BatchNorm1d(8*hidden_size),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(8*hidden_size, 4*hidden_size),
            nn.BatchNorm1d(4*hidden_size),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(4*hidden_size, 2*hidden_size),
            nn.BatchNorm1d(2*hidden_size),
            nn.Tanh(),
            nn.Linear(2*hidden_size, output_size)
        )
        # self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x1 = x[:, :self.input_size1]
        # x2 = x[:, self.input_size1:]
        x2 = x
        # 均值
        out1 = self.fc11(x1)
        # out1 = self.drop(self.fc11(x1))
        # 预测值+方差
        out2 = self.fc12(x2)
        # out2 = self.drop(self.fc12(x2))
        out = self.relu(self.norm(out1 + out2))
        # out = self.sigmoid(out1 + out2)
        out = self.fc2(out)
        # out = self.drop(self.fc2(out))
        return out
# 四层，不加dp，结果存一下
    

# 训练和验证模型
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience=50):
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.cuda(), batch_y.cuda()
            optimizer.zero_grad()
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.cuda(), batch_y.cuda()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        # 更新学习率
        scheduler.step(val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

    # 加载最佳模型
    model.load_state_dict(best_model_state)
    return model

# 测试模型
def test_model(model, test_loader):
    model.eval()
    predictions = []
    org_and_pred = []
    with torch.no_grad():
        for batch_X, y in test_loader:
            batch_X = batch_X.cuda().float()
            outputs = model(batch_X)
            predictions.extend(outputs.cpu().numpy())
            for org, pred in zip(y.cpu().numpy(), outputs.cpu().numpy()):
                # 'real', 'forecast'
                org_and_pred.append({'real': org, 'forecast': pred[0]})
    return np.array(predictions), org_and_pred

# 绘制预测值与真实值的折线图
def plot_predictions(y_true, y_pred, title):
    y_true = y_true*1e-9
    y_pred = y_pred*1e-9
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='True OT', color='blue')
    plt.plot(y_pred, label='Predicted OT', color='red', alpha=0.5)
    plt.text(0.5, 0.02, f'MSE: {mse:.9f}, MAE: {mae:.9f}', transform=plt.gca().transAxes, fontsize=12, horizontalalignment='center', verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    plt.xlabel('Time Steps')
    plt.ylabel('OT')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig("all_pred.png", dpi=300)
    plt.close()


def predict(save_path, model_path, data_path):
    X_train, X_test, y_train, y_test, data_mean, data_std, X, y = preprocess_data(data_path)
    test_dataset = CustomDataset(X, y)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    print(X[0].size)
    input_size = X[0].size
    input_size1 = 4
    input_size2 = 2
    hidden_size = 64
    output_size = 1
    model = GDNNOld(input_size1, input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()
    predictions, _ = test_model(model, test_loader)
    org_and_pred = []
    y = (y*data_std + data_mean)*1e-6
    y_pred = (predictions*data_std + data_mean)*1e-6
    for real, forecast in zip(y, y_pred):
        org_and_pred.append({'real': real, 'forecast': forecast[0]})
    with open(save_path, mode='w', newline= '') as csv_file:
        fieldnames = ['real', 'forecast']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in org_and_pred:
            writer.writerow(row)
    
    
    
    

# 主函数
def main():
    set_seed(2025)
    # 数据预处理
    # file_path1 = '/mnt/e/timer/PatchTST/offset/timestamp.csv'
    file_path1 = '/mnt/e/timer/PatchTST/offset/filtered_timestamp0118_with_pd_1.csv'
    # X_train, X_test, y_train, y_test, data_max, data_min, X, y = preprocess_data(file_path)
    # X_train, X_test, y_train, y_test, data_max, data_min, X, y = preprocess_data(file_path1)
    X_train, X_test, y_train, y_test, data_mean, data_std, X, y = preprocess_data(file_path1)
    # _, _, _, _, _, _, X, y = preprocess_data(file_path2)

    # 创建 Dataset 和 DataLoader
    # print(len(X_train), len(X_test), X_train[0].size(0))
    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_test, y_test)
    test_dataset = CustomDataset(X, y)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    # 初始化模型
    input_size = X_train[0].size(0)
    input_size1 = 4
    input_size2 = 2
    hidden_size = 64
    output_size = 1
    # model = GDNN(input_size1, input_size, hidden_size, output_size)
    model = GDNNOld(input_size1, input_size, hidden_size, output_size)
    # model = SimpleNN(input_size, hidden_size, output_size)
    # model = AddPowAmp(input_size1, input_size2, hidden_size, output_size)
    
    # model.load_state_dict(torch.load('model.pth'))
    # model.eval()
    model = model.cuda()


    # 定义损失函数和优化器
    # criterion = StdLoss()
    criterion = nn.HuberLoss()
    # criterion = nn.L1Loss()
    # criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 训练和验证模型
    num_epochs = 300
    model = train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs)

    # 测试模型
    predictions, _ = test_model(model, test_loader)

    y = y*data_std + data_mean
    y_pred = predictions*data_std + data_mean
    # y_test = y_test*data_std + data_mean
    # y_test = y_test.numpy()
    # y_pred = predictions*data_std + data_mean

    # 绘制预测值与真实值的折线图
    plot_predictions(y, y_pred, 'True vs Predicted OT on Test Set')

    # 保存模型
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved to 'model.pth'")

if __name__ == "__main__":
    main()
    predict(save_path='/mnt/e/timer/PatchTST/offset/org_and_ans_filtered.csv', model_path='model.pth', data_path='/mnt/e/timer/PatchTST/offset/filtered_timestamp0118_with_pd_1.csv')
    show_kalman_filter('/mnt/e/timer/PatchTST/offset/org_and_ans_filtered.csv', pic_path='/mnt/e/timer/PatchTST/offset/final_')
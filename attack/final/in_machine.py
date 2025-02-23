import pandas as pd
import numpy as np
import torch
import joblib
from collections import deque
from torch.utils.data import TensorDataset, DataLoader
import time

# ----------------------------
# 模型定义
# ----------------------------
class TemporalAttackDetector(torch.nn.Module):
    def __init__(self, input_size, num_classes=4):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, 64, num_layers=2, 
                                  batch_first=True, bidirectional=True)
        self.attention = torch.nn.MultiheadAttention(128, 4)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.GELU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.permute(1, 0, 2)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out[-1]
        return self.classifier(attn_out)

class AttackValuePredictor(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1)
        )
        self.branches = torch.nn.ModuleDict({
            str(i): torch.nn.Sequential(
                torch.nn.Linear(128, 64),
                torch.nn.GELU(),
                torch.nn.Linear(64, 1)
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

# ----------------------------
# 预测处理器
# ----------------------------
class SingleStepPredictor:
    def __init__(
        self,
        cls_model_path='cls_model_full_22.pth',
        reg_model_path='reg_model_full_22.pth',
        cls_scaler_path='cls_feature_scaler_22.pkl',
        reg_scaler_path='reg_feature_scaler_22.pkl',
        reg_target_scaler_path='reg_target_scaler_22.pkl',
        ):
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      
        # 加载标准化器
        self.cls_scaler = joblib.load(cls_scaler_path)
        self.reg_scaler = joblib.load(reg_scaler_path)
        self.target_scaler = joblib.load(reg_target_scaler_path)
      
        # 初始化分类模型
        self.cls_model = TemporalAttackDetector(
            input_size=self.cls_scaler.n_features_in_
        ).to(self.device)
        self.cls_model.load_state_dict(torch.load(cls_model_path, map_location=self.device))
        self.cls_model.eval()
      
        # 初始化回归模型
        self.reg_model = AttackValuePredictor(
            input_dim=self.reg_scaler.n_features_in_
        ).to(self.device)
        self.reg_model.load_state_dict(torch.load(reg_model_path, map_location=self.device))
        self.reg_model.eval()
      
        # 初始化数据窗口
        self.cls_window = deque(maxlen=20)  # 分类模型窗口大小
        self.reg_window = deque(maxlen=20)  # 回归模型窗口大小

    def _prepare_regression_features(self, attack_type):
        """生成与训练时一致的回归特征"""
        # 统计特征字段
        stat_cols = ['master_offset', 'path_delay', 's2_freq', 'adjusted_Offset']
      
        # 当前时间步特征字段
        current_cols = [
            'timestamp', 'master_offset', 'path_delay', 's2_freq',
            'adjusted_Offset', 'adjusted_Path_Delay', 'PTPts', 'attack_type'
        ]
      
        # 计算统计特征
        df_window = pd.DataFrame(self.reg_window)
        stats = []
        for col in stat_cols:
            stats += [
                df_window[col].mean(),
                df_window[col].std(),
                df_window[col].max() - df_window[col].min(),
                df_window[col].quantile(0.25)
            ]
      
        # 当前时间步特征
        current = [
            self.reg_window[-1]['timestamp'],
            self.reg_window[-1]['master_offset'],
            self.reg_window[-1]['path_delay'],
            self.reg_window[-1]['s2_freq'],
            self.reg_window[-1]['adjusted_Offset'],
            self.reg_window[-1]['adjusted_Path_Delay'],
            self.reg_window[-1]['PTPts'],
            attack_type
        ]
      
        return np.array(stats + current).reshape(1, -1)

    def predict(self, data):
        # 输入验证
        required_fields = {
            'timestamp': float,
            'master_offset': float,
            'path_delay': float,
            's2_freq': float,
            'adjusted_Offset': float,
            'adjusted_Path_Delay': float,
            'PTPts': float
        }
        try:
            input_data = {k: convert(data[k]) for k, convert in required_fields.items()}
        except KeyError as e:
            raise ValueError(f"缺少必要字段: {e}") from None
      
        # 更新分类窗口
        cls_features = [
            input_data['timestamp'],
            input_data['master_offset'],
            input_data['s2_freq'],
            input_data['path_delay'],
            input_data['adjusted_Offset'],
            input_data['adjusted_Path_Delay'],
            input_data['PTPts']
        ]
        self.cls_window.append(cls_features)
      
        # 更新回归窗口
        self.reg_window.append(input_data)
      
        # 初始化默认值
        attack_type = 0
        attack_value = 0.0
      
        # 分类预测
        if len(self.cls_window) >= 20:
            # 标准化处理
            scaled = self.cls_scaler.transform(np.array(self.cls_window))
            tensor_data = torch.FloatTensor(scaled).unsqueeze(0).to(self.device)

            # 模型预测
            with torch.no_grad():
                output = self.cls_model(tensor_data)
                attack_type = torch.argmax(output).item()
      
        # 回归预测
        if attack_type != 0 and len(self.reg_window) >= 20:
            try:
                # 生成特征
                features = self._prepare_regression_features(attack_type)
                X_reg = self.reg_scaler.transform(features[:, :-1])  # 排除attack_type列
              
                # 模型预测
                tensor_data = torch.FloatTensor(X_reg).to(self.device)
                tensor_types = torch.LongTensor([attack_type]).to(self.device)
              
                with torch.no_grad():
                    pred = self.reg_model(tensor_data, tensor_types)
                    attack_value = self.target_scaler.inverse_transform(
                        pred.cpu().numpy().reshape(-1, 1)
                    )[0][0]
            except Exception as e:
                print(f"回归预测异常: {str(e)}")
      
        return attack_type, attack_value

# ----------------------------
# CSV处理函数
# ----------------------------
def process_csv(input_path, output_path):
    predictor = SingleStepPredictor()
    results = []
    total_time = 0.0  # 总耗时
    count = 0  # 处理的数据条数
  
    # 分块读取大文件
    for chunk in pd.read_csv(input_path, chunksize=1000):
        for _, row in chunk.iterrows():
            try:
                start_time = time.time()  # 开始计时
                # 构建输入数据
                input_data = {
                    'timestamp': row['timestamp'],
                    'master_offset': row['master_offset'],
                    'path_delay': row['path_delay'],
                    's2_freq': row['s2_freq'],
                    'adjusted_Offset': row['adjusted_Offset'],
                    'adjusted_Path_Delay': row['adjusted_Path_Delay'],
                    'PTPts': row['PTPts']
                }
              
                # 执行预测
                pred_type, pred_value = predictor.predict(input_data)
              
                results.append({
                    **row.to_dict(),
                    'predicted_attack_type': pred_type,
                    'predicted_attack_value': pred_value
                })
                end_time = time.time()  # 结束计时
                elapsed_time = end_time - start_time  # 计算耗时
                total_time += elapsed_time  # 累加耗时
                count += 1  # 增加计数
            except Exception as e:
                print(f"处理第{_+1}行出错: {str(e)}")
    
    # 计算平均耗时
    average_time = total_time / count if count > 0 else 0.0
    print(f"平均每条数据处理耗时: {average_time:.6f} 秒")
  
    # 保存结果
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"预测结果已保存至 {output_path}")

# 使用示例
if __name__ == "__main__":
    process_csv("fixed_data_22.csv", "output_with_predictions22.csv")\

    """
    部署预测示例：
    输入数据示例：
    input_data = {
        'timestamp': row['timestamp'],
        'master_offset': row['master_offset'],
        'path_delay': row['path_delay'],
        's2_freq': row['s2_freq'],
        'adjusted_Offset': row['adjusted_Offset'],
        'adjusted_Path_Delay': row['adjusted_Path_Delay'],
        'PTPts': row['PTPts']
    }
    predictor = SingleStepPredictor()
    pred_type, pred_value = predictor.predict(input_data)
    # 补偿回offset与delay
    new_offset = input_data['adjusted_Offset'] - pred_value/2
    new_delay = input_data['adjusted_Path_Delay'] + pred_value/2
    """

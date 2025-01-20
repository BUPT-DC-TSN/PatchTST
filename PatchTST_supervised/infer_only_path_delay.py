import torch.utils
import torch.utils.data
from exp.exp_basic import Exp_Basic
from models import PatchTST
import argparse

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler


import numpy as np
import torch
import torch.nn as nn
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Dataset_Infer_No_ot(torch.utils.data.Dataset):
    def __init__(self, root_path, data_path, seq_len, pred_len):
        self.scale = StandardScaler()
        df_raw = pd.read_csv(os.path.join(root_path, data_path))
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)
        
        self.df_data = df_raw[border1:border2]
        self.df_data.reset_index(drop=True)
        
        tmp_stamp = df_raw[['date']][border1:border2]
        self.df_data['date'] = pd.to_datetime(tmp_stamp.date)
        self.df_data.date = list(tmp_stamp.date.values)
        self.df_data['date'] = pd.to_datetime(self.df_data.date)
        self._get_time_feature()
        cols = list(self.df_data.columns)
        cols.remove('Seconds_sin')
        cols.remove('Seconds_cos')
        cols.remove('master_offset')
        cols.remove('freq')
        cols.remove('OT')
        cols = ['Seconds_sin', 'Seconds_cos'] + cols + ['OT']
        # cols = ['Seconds_sin', 'Seconds_cos'] + cols
        self.df_data = self.df_data[cols]
        
        self.df_data_x = self.df_data.drop(['OT'], axis=1)
        self.data_y = self.df_data['OT']
        self.data_y = np.expand_dims(self.data_y, axis=-1)

        self.scale.fit(self.df_data_x.values)
        self.data_x = self.scale.transform(self.df_data_x)
        

        
    def _get_time_feature(self):
        self.df_data['hour'] = self.df_data['date'].dt.hour
        self.df_data['minute'] = self.df_data['date'].dt.minute
        self.df_data['seconds'] = self.df_data['date'].dt.second
        self.df_data['fractional_seconds'] = (self.df_data['date'] - self.df_data['date'].dt.floor('s')).dt.total_seconds()
        
        seconds_in_day = self.df_data['hour'] * 3600 + self.df_data['minute'] * 60 + self.df_data['seconds'] + self.df_data['fractional_seconds']
        day_seconds = 24 * 60 * 60
        # 计算正弦和余弦特征
        self.df_data['Seconds_sin'] = np.sin(2 * np.pi * seconds_in_day / day_seconds)
        self.df_data['Seconds_cos'] = np.cos(2 * np.pi * seconds_in_day / day_seconds)
        
        self.df_data = self.df_data.drop(['date', 'hour', 'minute', 'seconds', 'fractional_seconds'], axis=1)

        
                
    def _get_time_feature_single(self, date):
        date = pd.to_datetime(date)
        
        hour = date.hour
        minute = date.minute
        seconds = date.second
        fractional_seconds = (date - date.floor('s')).total_seconds()
        
        seconds_in_day = hour * 3600 + minute * 60 + seconds + fractional_seconds
        day_seconds = 24 * 60 * 60
        
        seconds_sin = np.sin(2 * np.pi * seconds_in_day / day_seconds)
        seconds_cos = np.cos(2 * np.pi * seconds_in_day / day_seconds)
        
        return {
            'Seconds_sin': seconds_sin,
            'Seconds_cos': seconds_cos
        }


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        seq_x = self.data_x[s_begin:s_end]
        
        return seq_x

    def update_data(self, path_delay, date):
        date_features = self._get_time_feature_single(date)
        new_data = {
            "path_delay": path_delay,
            **date_features,
        }
        self.df_data_x = self.df_data_x[1:]
        self.df_data_x = pd.concat([self.df_data_x, pd.DataFrame([new_data])], ignore_index=True)
        self.scale.fit(self.df_data_x.values)
        self.data_x = self.scale.transform(self.df_data_x.values)


    def inverse_output(self, pred):
        return pred * 1e6

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1


class Infer:
    def __init__(self, args, checkpoint_path, root_path, data_path, seq_len, pred_len, input_type, device):
        self.model = PatchTST.Model(args).float()
        self.model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'checkpoint.pth')))
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.pred_len = pred_len
        self.input_type = input_type
        
        self.pred_data = Dataset_Infer_No_ot(
            root_path=root_path,
            data_path=data_path,
            seq_len=seq_len,
            pred_len=pred_len
        )
        
        self.preds = []
        self.original_data = []
        
    def infer(self, pred_path=None):
        with torch.no_grad():
            input_x = self.pred_data.data_x
            input_x = np.expand_dims(input_x, axis=0)
            input_x = torch.from_numpy(input_x).float().to(self.device)
            outputs = self.model(input_x)
            pred = self._modify_outputs(outputs)
            self.preds.append(pred)
            if self.input_type == 0:
                # 手动输入(要真值)
                while True:
                    print(pred)
                    # === ===== ====测试输入数据(实际要换成数据接收程序)
                    data = input("输入 'q' 退出：")
                    if data.lower() == 'q':
                        break
                    try:
                        master_offset, freq, path_delay, date, ot = data.split(',')
                        master_offset = float(master_offset)
                        freq = float(freq)
                        path_delay = float(path_delay)
                        ot = float(ot)
                    except ValueError:
                        print("输入格式错误 master_offset,freq,path_delay,date,OT")
                        continue
                    
                    # ==== ==== ====

                    pred = self._data_infer(path_delay, date)
                    self.preds.append(pred)
            elif self.input_type == 1:
                # 文件读取(不给真值)
                assert pred_path != None, "file path is None!!"
                late_data = pd.read_csv(pred_path)
                for i in range(len(late_data)):
                    master_offset = late_data.iloc[i]['master_offset']
                    freq = late_data.iloc[i]['freq']
                    path_delay = late_data.iloc[i]['path_delay']
                    date = late_data.iloc[i]['date']
                    ot = late_data.iloc[i]['OT']
                    pred = self._data_infer(path_delay, date)
                    self.preds.append(pred)
                    self.original_data.append(ot)
                
            
            elif self.input_type == 2:
                # 文件读取(给真值)
                assert pred_path != None, "file path is None!!"
                late_data = pd.read_csv(pred_path)
                for i in range(len(late_data)):
                    master_offset = late_data.iloc[i]['master_offset']
                    freq = late_data.iloc[i]['freq']
                    path_delay = late_data.iloc[i]['path_delay']
                    date = late_data.iloc[i]['date']
                    ot = late_data.iloc[i]['OT']
                    pred = self._data_infer(path_delay, date)
                    self.preds.append(pred)
                    self.original_data.append(ot)
                    
            elif self.input_type == 3:
                assert pred_path is not None, "file path is None!!"
                late_data = pd.read_csv(pred_path)
                total_steps = len(late_data)  # 总时间步数
                pred_buffer = np.zeros((total_steps + self.pred_len, self.pred_len))  # 缓存每个位置的预测值
                pred_counts = np.zeros((total_steps + self.pred_len, 1))  # 记录每个位置被预测的次数

                for i in range(total_steps):
                    # 获取当前输入数据
                    master_offset = late_data.iloc[i]['master_offset']
                    freq = late_data.iloc[i]['freq']
                    path_delay = late_data.iloc[i]['path_delay']
                    date = late_data.iloc[i]['date']
                    ot = late_data.iloc[i]['OT']

                    # 更新数据集
                    self.pred_data.update_data(path_delay, date, ot)

                    # 进行预测
                    input_x = self.pred_data.data_x
                    input_x = np.expand_dims(input_x, axis=0)
                    input_x = torch.from_numpy(input_x).float().to(self.device)
                    outputs = self.model(input_x)
                    pred = self._modify_outputs(outputs)  # 获取预测值 (pred_len,)

                    # 将预测值存入缓存
                    for j in range(self.pred_len):
                        pred_buffer[i + j, j] += pred[j]  # 累加预测值
                        pred_counts[i + j] += 1  # 记录预测次数

                    # 取第一个预测值更新数据集（滑动窗口）
                    # self.pred_data.update_data(master_offset, freq, path_delay, date, pred[0])

                # 计算每个位置的预测均值
                pred_means = []
                for i in range(total_steps):
                    if pred_counts[i] > 0:
                        pred_means.append(pred_buffer[i, :self.pred_len].sum() / pred_counts[i])
                    else:
                        pred_means.append(0)  # 如果没有预测值，填充0

                # 保存预测结果
                self.preds = pred_buffer[:total_steps, 0]  # 只取第一个预测值作为最终预测结果
                # self.preds = pred_means
                self.original_data = late_data['OT'].values  # 保存真实值

                 
                pass
            # TODO: elif 实时部署
            
        self.preds = np.array(self.preds)
        self.original_data = np.array(self.original_data)

    
    def draw(self, save_path, pic_name="pred_and_original.png"):
        if self.input_type != 3:
            draw_preds = self.preds[:-1, :]
        else:
            draw_preds = self.preds
        draw_preds = draw_preds*1e-9
        original = self.original_data
        original = original*1e-9
        # 计算 MSE 和 MAE
        mse = mean_squared_error(original, draw_preds)
        mae = mean_absolute_error(original, draw_preds)
        
        # 绘制折线图
        plt.figure(figsize=(12, 6))
        plt.plot(original, label='Original Data', color='blue')
        plt.plot(draw_preds, label='Predicted Data', color='red')
        plt.text(0.5, 0.02, f'MSE: {mse:.9f}, MAE: {mae:.9f}', transform=plt.gca().transAxes, fontsize=12, horizontalalignment='center', verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.title('Original vs Predicted Values Over Time')
        plt.legend()
        plt.savefig(os.path.join(save_path, pic_name), dpi=300)
        
    
    def _data_infer(self, path_delay, date):
        self.pred_data.update_data(path_delay, date)
        input_x = self.pred_data.data_x
        input_x = np.expand_dims(input_x, axis=0)
        input_x = torch.from_numpy(input_x).float().to(self.device)
        outputs = self.model(input_x)
        pred = self._modify_outputs(outputs)
        return pred

    def _modify_outputs(self, outputs):
        pred = outputs.detach().cpu().numpy()
        pred = pred.squeeze(0)
        # pred = self.pred_data.scale.inverse_transform(pred)
        pred = self.pred_data.inverse_output(pred)
        pred = pred[-self.pred_len:, -1:]
        pred = pred.squeeze(1)
        
        return pred        


def main(args):
    #  args, checkpoint_path, root_path, data_path, seq_len, pred_len, input_type, device):
    infer_model = Infer(
        args=args,
        checkpoint_path=args.checkpoint_path,
        root_path=args.root_path,
        data_path=args.data_path,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        input_type=args.input_type,
        device=args.device
    )
    infer_model.infer(pred_path=args.pred_path)
    if args.draw == 1:
        infer_model.draw(save_path=args.save_path, pic_name=args.pic_name)

    # TODO: 保存或者输出结果
    print(infer_model.pred_data)
        

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Loacl Inference script for PatchTST')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')


    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # Formers 
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    parser.add_argument('--device', type=str, default='cuda', help='device ids of multile gpus')


    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--pred_path", type=str, default=None)
    parser.add_argument("--input_type", type=int, default=1)
    parser.add_argument("--draw", type=int, default=1)
    parser.add_argument("--pic_name", type=str, default="pred_and_original.png")

    args = parser.parse_args()
    main(args)
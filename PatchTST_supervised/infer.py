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

class Dataset_Infer(torch.utils.data.Dataset):
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
        cols.remove('OT')
        cols = ['Seconds_sin', 'Seconds_cos'] + cols + ['OT']
        self.df_data = self.df_data[cols]

        self.scale.fit(self.df_data.values)
        self.data_x = self.scale.transform(self.df_data)

        
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

    def update_data(self, master_offset, freq, path_delay, date, OT):
        date_features = self._get_time_feature_single(date)
        new_data = {
            "master_offset": master_offset,
            "freq": freq,
            "path_delay": path_delay,
            **date_features,
            "OT": OT,
        }
        self.df_data = self.df_data[1:]
        self.df_data = pd.concat([self.df_data, pd.DataFrame([new_data])], ignore_index=True)
        self.scale.fit(self.df_data.values)
        self.data_x = self.scale.transform(self.df_data)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1



def infer(args, checkpoint_path, root_path, data_path, seq_len, pred_len, device):
    model = PatchTST.Model(args).float()
    model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'checkpoint.pth')))
    model.to(device)
    model.eval()
    f_dim = -1 # 单特征输出只取最后一个特征
    
    pred_data = Dataset_Infer(
        root_path=root_path,
        data_path=data_path,
        seq_len=seq_len,
        pred_len=pred_len
    )
    
    preds = []
    
    with torch.no_grad():
        while True:
            input_x = pred_data.data_x
            input_x = np.expand_dims(input_x, axis=0)
            input_x = torch.from_numpy(input_x).float().to(device)
            outputs = model(input_x)
            pred = outputs.detach().cpu().numpy()
            pred = pred.squeeze(0)
            pred = pred_data.scale.inverse_transform(pred)
            pred = pred[-pred_len:, f_dim:]
            pred = pred.squeeze(0).squeeze(0)
            preds.append(pred)
            print(pred)
            
            # === ===== ====测试输入数据(实际要换成数据接收程序)
            data = input("输入 'q' 退出：")
            if data.lower() == 'q':
                break
            try:
                master_offset, freq, path_delay, date, OT = data.split(',')
                master_offset = float(master_offset)
                freq = float(freq)
                path_delay = float(path_delay)
                OT = float(OT)
            except ValueError:
                print("输入格式错误 master_offset,freq,path_delay,date,OT")
                continue
            
            # ==== ==== ====

            pred_data.update_data(master_offset, freq, path_delay, date, OT)
    
    return preds

def main(args):
    preds_list = infer(
        args,
        args.checkpoint_path,
        args.root_path,
        args.data_path,
        args.seq_len,
        args.pred_len,
        args.device
    )
    
    
    return preds_list
    


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

    args = parser.parse_args()
    main(args)
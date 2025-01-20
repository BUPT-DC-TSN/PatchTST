# 数据转换

`data_dir`为数据(命名为`logs.txt`以及`times.txt`)的目录

```python
cd dataset
python convert --data_date_path data_dir
```

转换好的数据在data_dir下

0112以及0113的wifi数据均在dataset目录下

# train

```bash
cd PatchTST_supervised/scripts/PatchTST/
bash train.sh
```

train.sh:

```python
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/wifi" ]; then
    mkdir ./logs/wifi
fi
seq_len=336
model_name=PatchTST

# root_path_name=./dataset/
root_path_name=/mnt/e/timer/dataset/transformer/0112
data_path_name=timestamp.csv
model_id_name=0112_wifi_2
data_name=custom
pred_len=1

random_seed=2021
python -u ../../run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --train_epochs 300\
    --patience 20\
    --lradj 'TST'\
    --pct_start 0.4 \
    --use_hour_sin 0 \
    --batch_size 256 --learning_rate 0.0002 >logs/wifi/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
```

参数解释:

- `root_path`数据集目录
- `data_path`数据集文件名(xx.csv)
- `data`目前私有数据集只能用custom

- `enc_in`输入特征数
- `seq_len`输入时间步长度
- `pred_len`预测时间步长度
- `features`: MS多特征输入单特征预测
- `patience`验证集损失不下降的轮次超过这个数就early stop
- `lradj`学习率调整方法，不用改
- `pct_start`学习率开始下降的轮次
- `use_hour_sin`是否添加时间戳作为特征0加1不加(这里时间戳会只保留时分秒并用三角函数编码)，不加就没时间特征输入模型

训练时会实时将每个轮次的信息保存在执行脚本目录下的`logs/wifi/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log`中



# predict/draw

```bash
cd PatchTST_supervised/scripts/PatchTST/
bash infer.sh
```

infer.sh:

```bash
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Our" ]; then
    mkdir ./logs/Our
fi
seq_len=336
model_name=PatchTST

# root_path_name=./dataset/
root_path_name=/mnt/e/timer/dataset/transformer/0112
data_path_name=predict.csv
model_id_name=0112_wifi
data_name=custom
pred_len=1

random_seed=2021
python -u ../../run_longExp.py \
    --random_seed $random_seed \
    --is_training 0 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name_$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --patch_len 16\
    --stride 8\
    --draw 1 \
    --use_hour_sin 0 \
    --checkpoint_path 'checkpoints/0112_wifi_2_easy_date_336_1_PatchTST_custom_ftMS_sl336_ll48_pl1_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0'\
    --batch_size 128
```

与`train.sh`类似

特殊的地方：

- `draw`: 1绘图 0预测单步，绘制后的图像会保存在`checkpoint.pth`同目录下
- `checkpoint_path` 模型权重目录(权重为目录下的`checkpoint.pth`)


01.18 update: 想要模型预测不逐步趋于定值，得输入上一步真实的te

01.19 update: 是模型没训好吗？时序是否必要？

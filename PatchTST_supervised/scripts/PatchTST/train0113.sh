if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/wifi" ]; then
    mkdir ./logs/wifi
fi
seq_len=336
model_name=PatchTST

# root_path_name=./dataset/
root_path_name=/home/test/Documents/PatchTST/dataset/0113
data_path_name=timestamp.csv
model_id_name=0113_wifi
data_name=custom
pred_len=1
# epoch 100 --> 10

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
    --patch_len 16 \
    --stride 8 \
    --train_epochs 300\
    --patience 20\
    --lradj 'TST'\
    --pct_start 0.4 \
    --use_hour_sin 1 \
    --batch_size 1024 --learning_rate 0.0002 >logs/wifi/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

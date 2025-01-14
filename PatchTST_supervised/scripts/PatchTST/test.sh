if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Our" ]; then
    mkdir ./logs/Our
fi
seq_len=336
model_name=PatchTST

checkpoint_path='/mnt/e/timer/PatchTST/PatchTST_supervised/scripts/PatchTST/checkpoints/336_1_PatchTST_custom_ftMS_sl336_ll48_pl1_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0'
root_path_name=/mnt/e/timer/dataset/transformer/0112
data_path_name=predict.csv
model_id_name=0112_wifi
data_name=custom
pred_len=1
# epoch 100 --> 10

random_seed=2021
python -u ../../run_longExp.py \
    --random_seed $random_seed \
    --is_training 2 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name_$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 9 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --patch_len 16\
    --stride 8\
    --checkpoint_path $checkpoint_path \
    --batch_size 256 
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Our" ]; then
    mkdir ./logs/Our
fi
seq_len=336
model_name=PatchTST

root_path_name=/home/hxli/Documents/PatchTST/dataset/0112
data_path_name=timestamp.csv
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
    --checkpoint_path '/home/hxli/Documents/PatchTST/PatchTST_supervised/scripts/PatchTST/checkpoints/0112_wifi_336_1_PatchTST_custom_featureMS_seqlen336_labellen48_predlen1_d_model128_numheads16_elayers3_dlayers1_dff256_fc1_time_emb0'\
    --batch_size 128 

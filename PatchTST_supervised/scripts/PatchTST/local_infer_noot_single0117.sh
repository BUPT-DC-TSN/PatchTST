seq_len=600
root_path_name=/mnt/e/timer/PatchTST/dataset/0117/pred_test/
data_path_name=timestamp.csv
pred_len=1
checkpoint_path=/mnt/e/timer/PatchTST/PatchTST_supervised/scripts/PatchTST/checkpoints/0117_5g_no_ot_600_1_PatchTST_5g_no_ot_featureMS_seqlen600_labellen48_predlen1_patch16_stride8_d_model128_numheads16_elayers6_dlayers1_dff256_fc1_time_emb0
random_seed=2025
pred_path=/mnt/e/timer/PatchTST/dataset/0117/pred_test/pred.csv

python -u ../../infer_no_ot.py \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 3 \
    --e_layers 6 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --patch_len 16\
    --stride 8\
    --checkpoint_path $checkpoint_path \
    --save_path $checkpoint_path \
    --pred_path $pred_path \
    --input_type 1 \
    --pic_name "pred_pred.png" \
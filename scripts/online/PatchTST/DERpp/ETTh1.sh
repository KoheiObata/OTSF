export CUDA_VISIBLE_DEVICES=1
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/online" ]; then
    mkdir ./logs/online
fi

seq_len=512
data=ETTh1
model_name=PatchTST
online_method=DERpp

for pred_len in 48
do
for learning_rate in 0.0001
do
for online_learning_rate in 0.00001
do
  filename=logs/online/$model_name'_'$online_method'_'$data'_'$pred_len'_onlinelr'$online_learning_rate.log
  save_filename=checkpoints/$online_method'_'$data'_'$pred_len
  python -u run.py \
    --dataset $data --border_type 'online' \
    --model $model_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --itr 1 --skip $filename --online_method $online_method \
    --pretrain \
    --pin_gpu True --reduce_bs False \
    --save_opt --only_test \
    --save $save_filename \
    --learning_rate $learning_rate \
    --val_online_lr \
    --lradj type3 \
    --online_learning_rate $online_learning_rate >> $filename 2>&1
done
done
done

# for pred_len in 96
# do
# for learning_rate in 0.0003
# do
# for online_learning_rate in 0.00001
# do
#   suffix='_lr'$learning_rate'_onlinelr'$online_learning_rate
#   filename=logs/online/$model_name'_'$online_method'_mid'$mid'_share_fulltune_'$tune_mode'_'$data'_'$pred_len'_btl'$btl$suffix.log
#   python -u run.py \
#     --dataset $data --border_type 'online' --batch_size 16 \
#     --model $model_name \
#     --seq_len $seq_len \
#     --pred_len $pred_len \
#     --tune_mode $tune_mode \
#     --concept_dim $mid \
#     --val_online_lr --diff_online_lr \
#     --itr 3 --skip $filename --pretrain \
#     --save_opt --online_method $online_method \
#     --bottleneck_dim $btl \
#     --online_learning_rate $online_learning_rate --only_test \
#     --learning_rate $learning_rate --lradj type3 >> $filename 2>&1
# done
# done
# done

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/online" ]; then
    mkdir ./logs/online
fi

seq_len=336
data=ETTh2
model_name=PatchTST
online_method=Online
decomposition=1

# for pred_len in 24 48 96
for pred_len in 48
do
for online_learning_rate in 0.00003
do
  filename=logs/online/$model_name'_dc'$decomposition'_'$online_method'_'$data'_'$pred_len'_onlinelr'$online_learning_rate.log2
  python -u run.py \
    --dataset $data --border_type 'online' \
    --model $model_name \
    --decomposition $decomposition \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --itr 1 --skip $filename --online_method $online_method \
    --pin_gpu True --reduce_bs False \
    --save_opt --only_test \
    --online_learning_rate $online_learning_rate >> $filename 2>&1
done
done
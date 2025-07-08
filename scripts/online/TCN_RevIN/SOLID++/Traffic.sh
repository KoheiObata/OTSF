if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/online" ]; then
    mkdir ./logs/online
fi

seq_len=60
data=Traffic
model_name=TCN
online_method=SOLID

for buffer_size in 500
do
for mini_batch in 5
do
for pred_len in 24 48 96
do
for online_learning_rate in 0.00003
do
  filename=logs/online/$model_name'_RevIN_'$online_method'whole_cont_'$data'_'$pred_len'_N'$buffer_size'_n'$mini_batch'_onlinelr'$online_learning_rate.log
  python -u run.py \
    --dataset $data --border_type 'online' \
    --model $model_name --normalization RevIN \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --itr 3 --skip $filename --checkpoints "" --online_method $online_method --whole_model --continual --pretrain \
    --pin_gpu True --reduce_bs False \
    --save_opt --only_test --buffer_size $buffer_size --mini_batch $mini_batch \
    --online_learning_rate $online_learning_rate >> $filename 2>&1
done
done
done
done
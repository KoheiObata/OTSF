if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/online" ]; then
    mkdir ./logs/online
fi

seq_len=512
data=ETTm1
model_name=PatchTST
train_epochs=100

# for pred_len in 24 48 96
for pred_len in 48
do
for learning_rate in 0.0001
do
  python -u run.py \
    --dataset $data --border_type 'online' \
    --model $model_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --itr 1 --only_test \
    --pin_gpu True --reduce_bs False \
    --save_opt \
    --batch_size 128 \
    --train_epochs $train_epochs \
    --patience 10 \
    --learning_rate $learning_rate > logs/online/$model_name'_'$data'_'$pred_len'_lr'$learning_rate.log 2>&1
done
done
# file setting
model_name=PatchTST
border_type="online3test"


# experimental setting
data_arg=${1:-"ETTh2"}
seq_len_arg=${2:-"336"}
pred_len_arg=${3:-"24"}
devices=${4:-0}

# Offline learning setting
learning_rate_arg=${5:-"0.0001"}
optim_arg=${6:-"Adam"}
batch_size_arg=${7:-"8"}

iterations_arg=${8:-3}

for data in $data_arg
do
for seq_len in $seq_len_arg
do
for pred_len in $pred_len_arg
do
for learning_rate in $learning_rate_arg
do
for optim in $optim_arg
do
for batch_size in $batch_size_arg
do
  headpath=./results/$border_type/$model_name/$data/$seq_len'_'$pred_len
  modelpath='Offline/lr'$learning_rate'_'$optim'_'$batch_size
  savepath=$headpath/$modelpath
  if [ ! -d "$savepath" ]; then
    mkdir -p $savepath
    mkdir -p $savepath/logs
  fi
  python -u run.py \
    --dataset $data \
    --border_type $border_type \
    --model $model_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --use_multi_gpu \
    --devices $devices \
    --learning_rate $learning_rate \
    --optim $optim \
    --batch_size $batch_size \
    --train_epochs 100 \
    --patience 10 \
    --itr $iterations_arg \
    --skip \
    --save_opt \
    --train --valid --test \
    --savepath $savepath \
    --enable_detailed_metrics \
    --save_prediction \
    >> $savepath/logs/experiment.log 2>&1
done
done
done
done
done
done
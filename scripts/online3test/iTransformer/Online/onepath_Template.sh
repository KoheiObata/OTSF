# file setting
model_name=iTransformer
online_method=Online
border_type="online3test"


# experimental setting
data_arg=${1:-"ETTh2"}
seq_len_arg=${2:-"336"}
pred_len_arg=${3:-"24"}
devices=${4:-0}

# online learning setting
online_learning_rate_arg=${5:-"0.01"}
optim_arg=${6:-"SGD"}

iterations_arg=${7:-1}


for data in $data_arg
do
for seq_len in $seq_len_arg
do
for pred_len in $pred_len_arg
do
for online_learning_rate in $online_learning_rate_arg
do
for optim in $optim_arg
do
  headpath=./results/$border_type/$model_name/$data/$seq_len'_'$pred_len
  modelpath=$online_method/'olr'$online_learning_rate'_'$optim
  savepath=$headpath/$modelpath
  if [ ! -d "$savepath" ]; then
    mkdir -p $savepath
    mkdir -p $savepath/logs
  fi
  python -u run.py \
    --dataset $data \
    --border_type $border_type \
    --model $model_name \
    --online_method $online_method \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --use_multi_gpu \
    --devices $devices \
    --online_learning_rate $online_learning_rate \
    --optim $optim \
    --itr $iterations_arg \
    --skip \
    --save_opt \
    --online_train --online_valid --online_test \
    --savepath $savepath \
    --enable_detailed_metrics \
    --save_prediction \
    >> $savepath/logs/experiment.log 2>&1
done
done
done
done
done
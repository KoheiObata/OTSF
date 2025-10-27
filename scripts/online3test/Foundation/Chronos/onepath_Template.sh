# file setting
model_name=Foundation
online_method=Chronos
border_type="online3test"


# experimental setting
data_arg=${1:-"ETTh2"}
seq_len_arg=${2:-"336"}
pred_len_arg=${3:-"24"}
devices=${4:-0}

# online learning setting

iterations_arg=${5:-1}


for data in $data_arg
do
for seq_len in $seq_len_arg
do
for pred_len in $pred_len_arg
do
  headpath=./results/$border_type/$model_name/$data/$seq_len'_'$pred_len
  modelpath=$online_method
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
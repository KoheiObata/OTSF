# file setting
model_name=LinearRLS
online_method=LinearRLS
border_type="online3test"


# experimental setting
data_arg=${1:-"ETTh2"}
seq_len_arg=${2:-"336"}
pred_len_arg=${3:-"24"}
devices=${4:-0}

# online learning setting
forget_factor_arg=${5:-"1.0"}
update_batch_arg=${6:-"single"}
initial_train_batch_arg=${7:-100}
update_interval_arg=${8:-1}

iterations_arg=${9:-1}


for data in $data_arg
do
for seq_len in $seq_len_arg
do
for pred_len in $pred_len_arg
do
for forget_factor in $forget_factor_arg
do
for update_batch in $update_batch_arg
do
for initial_train_batch in $initial_train_batch_arg
do
for update_interval in $update_interval_arg
do
  headpath=./results/$border_type/$model_name/$data/$seq_len'_'$pred_len
  modelpath=$online_method/'ff'$forget_factor'_'$update_batch'_ib'$initial_train_batch'_ui'$update_interval
  savepath=$headpath/$modelpath
  if [ ! -d "$savepath" ]; then
    mkdir -p $savepath
    mkdir -p $savepath/logs
  fi
  python -u run.py \
    --dataset $data \
    --border_type $border_type \
    --model $model_name \
    --revin 0 \
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
    --forget_factor $forget_factor \
    --update_batch $update_batch \
    --initial_train_batch $initial_train_batch \
    --update_interval $update_interval \
    >> $savepath/logs/experiment.log 2>&1
done
done
done
done
done
done
done
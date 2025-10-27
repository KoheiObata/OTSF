#!/bin/bash

# file setting
model_name=DLinear
experiment_name="Pretrain"
online_method="Online"
border_type="online3test"

# experimental setting
data_arg=${1:-"ETTh2"}
seq_len_arg=${2:-"336"}
pred_len_arg=${3:-"24"}
devices=${4:-0}

optim_arg=${5:-"Adam"}


for seq_len in $seq_len_arg
do
for pred_len in $pred_len_arg
do
for data in $data_arg
do
for optim in $optim_arg
do
	# search parameters
	learning_rate="0.01 0.001 0.0001 0.00001"
	online_learning_rate="0.01 0.001 0.0001 0.00001"
	# Fixed parameters
	batch_size="8"

	# 1. Offline
	iterations="1"
	echo "--- Running Offline ---"
	echo "data: $data"
	echo "seq_len: $seq_len"
	echo "pred_len: $pred_len"
	echo "devices: $devices"
	echo "learning_rate: $learning_rate"
	echo "optim: $optim"
	echo "iterations: $iterations"
	echo "--------------------------------"
	sh scripts/online3test/DLinear/Offline/offline_Template.sh "$data" "$seq_len" "$pred_len" "$devices" "$learning_rate" "$optim" "$batch_size" "$iterations"

	# 2. Pretrain
	echo "--- Running Pretrain ---"
	echo "data: $data"
	echo "seq_len: $seq_len"
	echo "pred_len: $pred_len"
	echo "devices: $devices"
	echo "learning_rate: $learning_rate"
	echo "online_learning_rate: $online_learning_rate"
	echo "optim: $optim"
	echo "iterations: $iterations"
	echo "--------------------------------"
	sh scripts/online3test/DLinear/Online/validation_pretrain_Template.sh "$data" "$seq_len" "$pred_len" "$devices" "$learning_rate" "$optim" "$batch_size" "$online_learning_rate" "$iterations"


	# 3. Select the best setting in the search range.
	echo -e "\n--- Searching for Best Hyperparameters ---"
	echo "search_range"
	echo "learning_rate: $learning_rate"
	echo "online_learning_rate: $online_learning_rate"
	echo "batch_size: $batch_size"
	echo "optim: $optim"
	echo "--------------------------------"
	best_params_str=$(python scripts/hyperparameter_search.py \
		"$experiment_name" "$model_name" "$online_method" "$border_type" "$data" "$seq_len" "$pred_len" \
		--search_params "learning_rate=$learning_rate" "online_learning_rate=$online_learning_rate" "batch_size=$batch_size" "optim=$optim" \
		--return_param learning_rate online_learning_rate \
		)

	# check error
	if [ -z "$best_params_str" ]; then
		echo "Error: Failed to get the best learning rate. Exiting."
		exit 1
	fi

	# read command to assign space-separated string to each variable
	read -r best_lr best_olr <<< "$best_params_str"
	echo "Best Learning Rate found: $best_lr"
	echo "Best Online Learning Rate found: $best_olr"

	# 4. Run 3 experiments with the best setting.
	final_iterations="3"
	echo "--- Running Final Offline for Pretrain ---"
	echo "data: $data"
	echo "seq_len: $seq_len"
	echo "pred_len: $pred_len"
	echo "devices: $devices"
	echo "learning_rate: $best_lr"
	echo "optim: $optim"
	echo "iterations: $final_iterations"
	echo "--------------------------------"
	sh scripts/online3test/DLinear/Offline/offline_Template.sh "$data" "$seq_len" "$pred_len" "$devices" "$best_lr" "$optim" "$batch_size" "$final_iterations"

	# 5. Select the best setting and run 3 experiments.
	echo -e "\n--- Running Final Experiment ---"
	echo "data: $data"
	echo "seq_len: $seq_len"
	echo "pred_len: $pred_len"
	echo "devices: $devices"
	echo "learning_rate: $best_lr"
	echo "online_learning_rate: $best_olr"
	echo "batch_size: $batch_size"
	echo "optim: $optim"
	echo "iterations: $final_iterations"
	echo "--------------------------------"
	sh scripts/online3test/DLinear/Online/pretrain_Template.sh "$data" "$seq_len" "$pred_len" "$devices" "$best_lr" "$optim" "$batch_size" "$best_olr" "$final_iterations"
done
done
done
done
#!/bin/bash

# file setting
model_name=iTransformer
experiment_name="Online"
online_method=EWC
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
	online_learning_rate="0.01 0.001 0.0001 0.00001"
	buffer_size="64"
	mini_batch="4"
	ewc_lambda="1.0 10.0"

	# 1. Validation
	iterations="1"
	echo "--- Running Validation ---"
	echo "data: $data"
	echo "seq_len: $seq_len"
	echo "pred_len: $pred_len"
	echo "devices: $devices"
	echo "online_learning_rate: $online_learning_rate"
	echo "optim: $optim"
	echo "buffer_size: $buffer_size"
	echo "mini_batch: $mini_batch"
	echo "ewc_lambda: $ewc_lambda"
	echo "iterations: $iterations"
	echo "--------------------------------"
	sh scripts/online3test/iTransformer/EWC/validation_onepath_Template.sh "$data" "$seq_len" "$pred_len" "$devices" "$online_learning_rate" "$optim" "$buffer_size" "$mini_batch" "$ewc_lambda" "$iterations"


	# 2. find the best setting in search range
	echo -e "\n--- Searching for Best Hyperparameters ---"
	echo "search_range"
	echo "online_learning_rate: $online_learning_rate"
	echo "optim: $optim"
	echo "--------------------------------"
	best_params_str=$(python scripts/hyperparameter_search.py \
		"$experiment_name" "$model_name" "$online_method" "$border_type" "$data" "$seq_len" "$pred_len" \
		--search_params "online_learning_rate=$online_learning_rate" "optim=$optim" "buffer_size=$buffer_size" "mini_batch=$mini_batch" "ewc_lambda=$ewc_lambda" \
		--return_param online_learning_rate ewc_lambda \
		)

	# check error
	if [ -z "$best_params_str" ]; then
		echo "Error: Failed to get the best learning rate. Exiting."
		exit 1
	fi

	# read command to assign space-separated string to each variable
	read -r best_olr best_ewc_lambda <<< "$best_params_str"
	echo "Best Online Learning Rate found: $best_olr"
	echo "Best EWC Lambda found: $best_ewc_lambda"

	# 3. select the best setting and run 3 experiments
	final_iterations="3"
	echo -e "\n--- Running Final Experiment ---"
	echo "data: $data"
	echo "seq_len: $seq_len"
	echo "pred_len: $pred_len"
	echo "devices: $devices"
	echo "online_learning_rate: $best_olr"
	echo "optim: $optim"
	echo "buffer_size: $buffer_size"
	echo "mini_batch: $mini_batch"
	echo "ewc_lambda: $best_ewc_lambda"
	echo "iterations: $final_iterations"
	echo "--------------------------------"
	sh scripts/online3test/iTransformer/EWC/onepath_Template.sh "$data" "$seq_len" "$pred_len" "$devices" "$best_olr" "$optim" "$buffer_size" "$mini_batch" "$best_ewc_lambda" "$final_iterations"
done
done
done
done

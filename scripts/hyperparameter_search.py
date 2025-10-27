# hyperparameter_search.py (Robust Version)


import argparse
import json
import os
import itertools
import sys
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# --- Configuration items ---

# 1. Base directory where results are stored
BASE_RESULTS_DIR = './results'

# 2. Default evaluation metric and direction for optimization
DEFAULT_METRIC = 'mse'
# 'min' (minimize) or 'max' (maximize)
DEFAULT_OPTIM_DIRECTION = 'min'

# --- End of configuration items ---


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Find the best hyperparameters from validation results.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # --- Fixed arguments ---
    parser.add_argument('experiment_name', type=str, help="Experiment name (e.g., PatchTST)")
    parser.add_argument('model_name', type=str, help="Model name (e.g., PatchTST)")
    parser.add_argument('online_method', type=str, help="Online method (e.g., Online)")
    parser.add_argument('border_type', type=str, help="Directory group (e.g., online3test)")
    parser.add_argument('data', type=str, help="Dataset name (e.g., ETTh2)")
    parser.add_argument('seq_len', type=str, help="Sequence length")
    parser.add_argument('pred_len', type=str, help="Prediction length")

    # --- Search arguments ---
    parser.add_argument(
        '--search_params',
        nargs='+',
        required=True,
        help='''Space-separated list of hyperparameters to search.
            Format: param_name="value1 value2 ..."
            Example: --search_params online_learning_rate="0.01 0.001" optim="Adam SGD"'''
    )

    # --- Optional arguments ---
    parser.add_argument(
        '--return_param',
        nargs='+',  # <--- Change: accept multiple arguments
        required=True,
        help='''One or more names of hyperparameters to print to stdout.
            Values will be space-separated.
            Example: --return_param online_learning_rate optim'''
    )
    parser.add_argument(
        '--metric',
        type=str,
        default=DEFAULT_METRIC,
        help=f"Metric to optimize (default: {DEFAULT_METRIC})"
    )
    parser.add_argument(
        '--direction',
        type=str,
        default=DEFAULT_OPTIM_DIRECTION,
        choices=['min', 'max'],
        help=f"Optimization direction: 'min' or 'max' (default: {DEFAULT_OPTIM_DIRECTION})"
    )

    args = parser.parse_args()
    return args

def process_search_params(search_params_list: List[str]) -> Dict[str, List[str]]:
    """Convert --search_params argument list to dictionary format."""
    search_space = {}
    for item in search_params_list:
        try:
            key, value_str = item.split('=', 1)
            values = value_str.strip('"\'').split()
            search_space[key] = values
        except ValueError:
            print(f"Error: Invalid format for --search_params. Use 'key=\"value1 ...\"'. Problematic item: {item}", file=sys.stderr)
            sys.exit(1)
    return search_space


def construct_results_path(args: argparse.Namespace, param_combination: Dict[str, str]) -> str:
    """
    Construct the path to the JSON file of experiment results dynamically.
    """
    headpath=f"{args.border_type}/{args.model_name}/{args.data}/{args.seq_len}_{args.pred_len}"
    if args.experiment_name == "Online":
        if args.online_method == "Online":
            modelpath=f'validation/{args.online_method}/olr{param_combination["online_learning_rate"]}_{param_combination["optim"]}'
        elif args.online_method == "ER":
            modelpath=f'validation/{args.online_method}/olr{param_combination["online_learning_rate"]}_{param_combination["optim"]}/{param_combination["buffer_size"]}_{param_combination["mini_batch"]}_{param_combination["er_alpha"]}'
        elif args.online_method == "EWC":
            modelpath=f'validation/{args.online_method}/olr{param_combination["online_learning_rate"]}_{param_combination["optim"]}/{param_combination["buffer_size"]}_{param_combination["mini_batch"]}_{param_combination["ewc_lambda"]}'
    elif args.experiment_name == "Offline":
        modelpath=f'Offline/lr{param_combination["learning_rate"]}_{param_combination["optim"]}_{param_combination["batch_size"]}'
    elif args.experiment_name == "Pretrain":
        modelpath=f'validation/pretrain_{args.online_method}/lr{param_combination["learning_rate"]}_{param_combination["optim"]}_{param_combination["batch_size"]}/olr{param_combination["online_learning_rate"]}_{param_combination["optim"]}'
    else:
        modelpath=f'{args.online_method}'

    return os.path.join(BASE_RESULTS_DIR, headpath, modelpath, 'results.json')

def remove_duplicates(data):
    """
    Function to remove duplicates of 'seed' key in dictionary and also remove corresponding 'mse' and 'mae' values.
    Example:
    Original dict: {'mse': [0.15, 0.15, 0.11], 'mae': [0.25, 0.25, 0.22], 'seed': [2025, 2025, 2026]}
    Dict with duplicates removed: {'mse': [0.15, 0.11], 'mae': [0.25, 0.22], 'seed': [2025, 2026]}

    Args:
        data (dict): Dictionary with 'mse', 'mae', 'seed' keys.
                     Value of each key is a list, and number of elements must match.

    Returns:
        dict: New dictionary with unique 'seed' values.
    """
    # Initialize dictionary to map seeds to their indices
    seen_seeds = {}

    # Initialize lists to store results
    unique_mse = []
    unique_mae = []
    unique_seed = []

    # Loop through dictionary elements
    for i in range(len(data['seed'])):
        current_seed = data['seed'][i]

        # If current seed hasn't been found yet
        if current_seed not in seen_seeds:
            # Record unique seed and its index
            seen_seeds[current_seed] = i

            # Add corresponding mse and mae values to result lists
            unique_mse.append(data['mse'][i])
            unique_mae.append(data['mae'][i])
            unique_seed.append(current_seed)

    # Construct and return new dictionary
    return {'mse': unique_mse, 'mae': unique_mae, 'seed': unique_seed}

def get_valid_result(results_path: str, args: argparse.Namespace) -> float:
    """
    Get results from specified path.
    """
    with open(results_path, 'r') as f:
        results_data = json.load(f)

    if args.experiment_name == 'Offline':
        phase_name = f'valid_offline'
    elif args.experiment_name in ['Pretrain', 'Online']:
        phase_name = f'valid_online'

    results_dict = results_data['raw_results'][phase_name]
    results_dict = remove_duplicates(results_dict)
    return np.mean(results_dict[args.metric])

def find_best_hyperparameters(args: argparse.Namespace, search_space: Dict[str, List[str]]) -> Optional[Dict[str, Any]]:
    """Find the best hyperparameter combination from the specified search space."""
    best_metric = float('inf') if args.direction == 'min' else float('-inf')
    best_params = None

    param_names = list(search_space.keys())
    value_combinations = list(itertools.product(*search_space.values()))

    print(f"--- Hyperparameter Search Start ({len(value_combinations)} combinations) ---", file=sys.stderr)
    print(f"Optimizing for: {args.direction.upper()}({args.metric})", file=sys.stderr)

    for values in value_combinations:
        # Create current parameter combination as dictionary
        current_params = dict(zip(param_names, values))
        results_path = construct_results_path(args, current_params)

        if not os.path.exists(results_path):
            print(f"Info: File not found, skipping: {results_path}", file=sys.stderr)
            continue

        try:
            current_metric = get_valid_result(results_path, args)
            if current_metric is None:
                print(f"Info: No valid result found, skipping: {results_path}", file=sys.stderr)
                continue

            # Compare according to optimization direction
            is_better = (args.direction == 'min' and current_metric < best_metric) or \
                        (args.direction == 'max' and current_metric > best_metric)

            if is_better:
                best_metric = current_metric
                best_params = current_params
                print(f"  -> New best found: {current_params} | {args.metric}={best_metric:.6f}", file=sys.stderr)

        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read or parse {results_path}. Details: {e}", file=sys.stderr)
            continue

    print("--- Hyperparameter Search End ---", file=sys.stderr)
    return best_params

def main():
    """Main execution function."""
    args = parse_arguments()
    search_space = process_search_params(args.search_params)

    for param_key in args.return_param:
        if param_key not in search_space:
            print(f"Error: --return_param '{param_key}' is not one of the search parameters.", file=sys.stderr)
            sys.exit(1)

    best_params = find_best_hyperparameters(args, search_space)

    if best_params:
        print(f"\nOptimal configuration: {best_params}", file=sys.stderr)
        # Output optimal values of specified parameters to stdout
        output_values = [str(best_params[key]) for key in args.return_param]
        # Output as space-separated string to stdout
        print(" ".join(output_values))
    else:
        print("\nError: Could not determine best hyperparameters. No valid result files found.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
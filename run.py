#!/usr/bin/env python3

import argparse
import datetime
import gc
import os
import platform
import random
from pprint import pprint
import json

import numpy as np
import torch
import torch.distributed as dist

import settings
from data_provider import data_loader
from exp import *
import exp as exps
from exp.exp_online import Exp_Online
from settings import data_settings


def str_to_bool(value):
    """Helper function to convert string to boolean value"""
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def setup_argument_parser():
    """Create a structured argument parser"""
    parser = argparse.ArgumentParser(description='Online Time Series Forecasting')

    # =====================================
    # Basic Configuration
    # =====================================
    basic_group = parser.add_argument_group('Basic Configuration')
    basic_group.add_argument('--train', action='store_true', default=False, help='perform training')
    basic_group.add_argument('--valid', action='store_true', default=False, help='perform validation during training')
    basic_group.add_argument('--test', action='store_true', default=False, help='perform test after training')

    basic_group.add_argument('--override_hyper', action='store_true', default=True, help='Override hyperparams by setting.py')
    basic_group.add_argument('--compile', action='store_true', default=False, help='Compile the model by Pytorch 2.0')
    basic_group.add_argument('--reduce_bs', type=str_to_bool, default=False, help='Override batch_size in hyperparams by setting.py')
    basic_group.add_argument('--model', type=str, required=True, default='PatchTST')
    basic_group.add_argument('--normalization', type=str, default=None)
    basic_group.add_argument('--skip', action='store_true', default=False, help='skip the experiment')
    basic_group.add_argument('--savepath', type=str, default='./results/', help='location of results')

    # =====================================
    # Online Learning Configuration
    # =====================================
    online_group = parser.add_argument_group('Online Learning')
    online_group.add_argument('--checkpoints', type=str, default=None, help='location of model checkpoints')
    online_group.add_argument('--online_train', action='store_true', default=False, help='perform online learning using train set')
    online_group.add_argument('--online_valid', action='store_true', default=False, help='perform online learning using valid set')
    online_group.add_argument('--online_test', action='store_true', default=False, help='perform online learning using test set')

    online_group.add_argument('--online_method', type=str, default=None)
    online_group.add_argument('--online_learning_rate', type=float, default=None)
    online_group.add_argument('--save_opt', action='store_true', default=True)
    online_group.add_argument('--leakage', action='store_true', default=False)
    online_group.add_argument('--freeze', action='store_true', default=False)
    online_group.add_argument('--enable_detailed_metrics', action='store_true', default=False, help='enable detailed metrics collection during online learning')
    online_group.add_argument('--save_prediction', action='store_true', default=False, help='save predictions for each timestep during online learning')

    # =====================================
    # Proceed Method Configuration
    # =====================================
    proceed_group = parser.add_argument_group('Proceed Method')
    proceed_group.add_argument('--act', type=str, default='sigmoid', help='activation')
    proceed_group.add_argument('--tune_mode', type=str, default='down_up')
    proceed_group.add_argument('--ema', type=float, default=0, help='')
    proceed_group.add_argument('--concept_dim', type=int, default=200)
    proceed_group.add_argument('--bottleneck_dim', type=int, default=32, help='')
    proceed_group.add_argument('--individual_generator', action='store_true', default=False)
    proceed_group.add_argument('--share_encoder', action='store_true', default=False)
    proceed_group.add_argument('--use_mean', type=str_to_bool, default=True)
    proceed_group.add_argument('--joint_update_valid', action='store_true', default=False)
    proceed_group.add_argument('--comment', type=str, default='')
    proceed_group.add_argument('--wo_clip', action='store_true', default=False)

    # =====================================
    # ER, DERpp Method Configuration
    # =====================================
    er_group = parser.add_argument_group('ER, DERpp Method')
    er_group.add_argument('--ER_alpha', type=float, default=0.2, help='weight for ER and DERpp')
    er_group.add_argument('--ER_beta', type=float, default=0.2, help='weight for DERpp')

    # =====================================
    # EWC Method Configuration
    # =====================================
    ewc_group = parser.add_argument_group('EWC Method')
    ewc_group.add_argument('--ewc_lambda', type=float, default=1.0, help='weight for EWC')

    # =====================================
    # OneNet Method Configuration
    # =====================================
    onenet_group = parser.add_argument_group('OneNet Method')
    onenet_group.add_argument('--learning_rate_w', type=float, default=0.001, help='optimizer learning rate')
    onenet_group.add_argument('--learning_rate_bias', type=float, default=0.001, help='optimizer learning rate')

    # =====================================
    # SOLID Method Configuration
    # =====================================
    solid_group = parser.add_argument_group('SOLID Method')
    solid_group.add_argument('--lambda_period', type=float, default=0.1) # threshold for periodicity
    solid_group.add_argument('--whole_model', action='store_true') # set True to update entire model (freeze=False)
    solid_group.add_argument('--continual', action='store_true')

    # =====================================
    # BTOA Method Configuration
    # =====================================
    btoa_group = parser.add_argument_group('BTOA Method')
    btoa_group.add_argument('--savevae', default='ETTh1', type=str)
    btoa_group.add_argument('-dist', default='normal', type=str, choices=['normal', 'laplace', 'flow'])
    btoa_group.add_argument('-n', '--num-epochs', default=100, type=int, help='number of training epochs')
    btoa_group.add_argument('--VAE_learning_rate', default=2e-3, type=float, help='VAE learning rate')
    btoa_group.add_argument('-z', '--latent_dim', default=10, type=int, help='size of latent dimension')
    btoa_group.add_argument('--beta', default=5, type=float, help='ELBO penalty term')
    btoa_group.add_argument('--tcvae', action='store_true')
    btoa_group.add_argument('--exclude-mutinfo', action='store_true')
    btoa_group.add_argument('--beta-anneal', action='store_true')
    btoa_group.add_argument('--lambda-anneal', action='store_true')
    btoa_group.add_argument('--mss', action='store_true', help='use the improved minibatch estimator')
    btoa_group.add_argument('--conv', action='store_true')
    btoa_group.add_argument('--log_freq', default=200, type=int, help='num iterations per log')
    btoa_group.add_argument('--mean', type=float, default=0.9, help='Mean of Gaussian')
    btoa_group.add_argument('--std', type=float, default=0.1, help='std of Gaussian')
    btoa_group.add_argument('--low_limit', type=float, default=0.7, help='low limit of Gaussian')
    btoa_group.add_argument('--high_limit', type=float, default=1, help='high limit of Gaussian')
    btoa_group.add_argument('--aug1', type=str, default='na', choices=['na', 'noise', 'scale', 'negate', 'perm', 'shuffle', 't_flip', 't_warp', 'resample', 'random_out','rotation', 'perm_jit', 'jit_scal', 'hfc', 'lfc', 'p_shift', 'ap_p', 'ap_f'], help='the type of augmentation transformation')
    btoa_group.add_argument('--aug2', type=str, default='resample', choices=['na', 'noise', 'scale', 'negate', 'perm', 'shuffle', 't_flip', 't_warp', 'resample', 'random_out', 'rotation', 'perm_jit', 'jit_scal', 'hfc', 'lfc', 'p_shift', 'ap_p', 'ap_f'], help='the type of augmentation transformation')

    # =====================================
    # Proposed Method Configuration
    # =====================================
    proposed_group = parser.add_argument_group('Proposed Method')
    proposed_group.add_argument('--aug_method', type=str, default='mixbuff', choices=['mixbuff', 'nothing', 'noise', 'noisehalf', 'fft', 'btoa', 'mixseq'])
    proposed_group.add_argument('--max_lag', type=str, default='-period')
    proposed_group.add_argument('--min_lag', type=str, default='-seq_len+1')
    proposed_group.add_argument('--base_ratio_start', type=float, default=0.5)
    proposed_group.add_argument('--base_ratio_end', type=float, default=0.5)
    proposed_group.add_argument('--base_seq', type=str, default='buff', choices=['buff', 'seq'])
    proposed_group.add_argument('--detrend', action='store_true') # set True to update entire model (freeze=False)
    proposed_group.add_argument('--add_recent', type=str_to_bool, default=False)
    proposed_group.add_argument('--period', default=None, type=int, help='period')
    proposed_group.add_argument('--aug_ratio', type=float, default=0.5)

    # =====================================
    # RLS Method Configuration
    # =====================================
    rls_group = parser.add_argument_group('RLS')
    rls_group.add_argument('--forget_factor', type=float, default=0.99)
    rls_group.add_argument('--update_batch', type=str, default='single', choices=['multiple', 'single'])
    rls_group.add_argument('--initial_train_batch', default=100, type=int, help='initial_train_batch')
    rls_group.add_argument('--update_interval', default=1, type=int, help='update_interval')
    rls_group.add_argument('--predictor_label_path', type=str, default=None, help='location of predictor label')

    # =====================================
    # Power Method Configuration
    # =====================================
    power_group = parser.add_argument_group('Power')
    power_group.add_argument('--far_forget_rate', type=float, default=1.00)
    power_group.add_argument('--far_lambda_0', type=float, default=1.00)
    power_group.add_argument('--fft_bins', type=int, default=14)
    power_group.add_argument('--power_norm_method', type=str, default='zscore', choices=['l1', 'l2', 'minmax', 'zscore', 'log', 'none'])
    power_group.add_argument('--target_modules', type=str, default='all', help='comma-separated list of target module names (e.g., "module.model.head.linear,linear")')
    power_group.add_argument('--freeze_untarget', action='store_true', default=False, help='freeze untargeted parameters')
    power_group.add_argument('--grad_scale', type=str, default='direct', choices=['direct', 'normalize', 'softmax', 'stepnorm', 'negative'])
    power_group.add_argument('--update_affinity_after', action='store_true', default=False, help='update affinity matrix')
    power_group.add_argument('--new_optimizer', action='store_true', default=False, help='use new optimizer')
    power_group.add_argument('--noisetrain', type=int, default=0)

    # =====================================
    # Buffer Configuration (SOLIDpp, BTOA, ER, DERpp, DSOF, Proposed)
    # =====================================
    buffer_group = parser.add_argument_group('Buffer Configuration')
    buffer_group.add_argument('--mini_batch', type=int, default=5, help='mini_batch')
    buffer_group.add_argument('--buffer_size', type=int, default=100, help='buffer_size')

    # =====================================
    # Data Configuration
    # =====================================
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--border_type', type=str, default='online', help='set any other value for traditional data splits')
    data_group.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    data_group.add_argument('--dataset', type=str, default='ETTh1', help='data file')
    data_group.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    data_group.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    data_group.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    data_group.add_argument('--use_time', action='store_true', default=False, help='use time features or not')

    # =====================================
    # Forecasting Task Configuration
    # =====================================
    forecasting_group = parser.add_argument_group('Forecasting Task')
    forecasting_group.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    forecasting_group.add_argument('--label_len', type=int, default=48, help='start token length')
    forecasting_group.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    forecasting_group.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

    # =====================================
    # PatchTST Configuration
    # =====================================
    patchtst_group = parser.add_argument_group('PatchTST Configuration')
    patchtst_group.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    patchtst_group.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    patchtst_group.add_argument('--patch_len', type=int, default=16, help='patch length')
    patchtst_group.add_argument('--stride', type=int, default=8, help='stride')
    patchtst_group.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    patchtst_group.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    patchtst_group.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    patchtst_group.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    patchtst_group.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    patchtst_group.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    patchtst_group.add_argument('--drop_last', action='store_true', default=False)

    # =====================================
    # Transformer Configuration
    # =====================================
    transformer_group = parser.add_argument_group('Transformer Configuration')
    transformer_group.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    transformer_group.add_argument('--d_model', type=int, default=512, help='dimension of model')
    transformer_group.add_argument('--n_heads', type=int, default=8, help='num of heads')
    transformer_group.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    transformer_group.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    transformer_group.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    transformer_group.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    transformer_group.add_argument('--factor', type=int, default=3, help='attn factor')
    transformer_group.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
    transformer_group.add_argument('--dropout', type=float, default=0.05, help='dropout')
    transformer_group.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    transformer_group.add_argument('--activation', type=str, default='gelu', help='activation')
    transformer_group.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    transformer_group.add_argument('--output_enc', action='store_true', help='whether to output embedding from encoder')
    transformer_group.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # =====================================
    # Crossformer Configuration
    # =====================================
    crossformer_group = parser.add_argument_group('Crossformer Configuration')
    crossformer_group.add_argument('--seg_len', type=int, default=24, help='segment length (L_seg)')
    crossformer_group.add_argument('--win_size', type=int, default=2, help='window size for segment merge')
    crossformer_group.add_argument('--num_routers', type=int, default=10, help='num of routers in Cross-Dimension Stage of TSA (c)')

    # =====================================
    # Other Model Configuration
    # =====================================
    other_models_group = parser.add_argument_group('Other Model Configuration')
    other_models_group.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    other_models_group.add_argument('--subgraph_size', type=int, default=20, help='k')
    other_models_group.add_argument('--in_dim', type=int, default=1)
    other_models_group.add_argument('--gpt_layers', type=int, default=6)
    other_models_group.add_argument('--tmax', type=int, default=10)
    other_models_group.add_argument('--patch_size', type=int, default=16)

    # =====================================
    # Optimization Configuration
    # =====================================
    optimization_group = parser.add_argument_group('Optimization')
    optimization_group.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    optimization_group.add_argument('--itr', type=int, default=5, help='experiments times')
    optimization_group.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    optimization_group.add_argument('--begin_valid_epoch', type=int, default=0)
    optimization_group.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    optimization_group.add_argument('--patience', type=int, default=5, help='early stopping patience')
    optimization_group.add_argument('--optim', type=str, default='Adam')
    optimization_group.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    optimization_group.add_argument('--des', type=str, default='test', help='exp description')
    optimization_group.add_argument('--loss', type=str, default='mse', help='loss function')
    optimization_group.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    optimization_group.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    optimization_group.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    optimization_group.add_argument('--warmup_epochs', type=int, default=5)

    # =====================================
    # GPU Configuration
    # =====================================
    gpu_group = parser.add_argument_group('GPU Configuration')
    gpu_group.add_argument('--use_gpu', type=str_to_bool, default=True, help='specify whether to use GPU (True/False)')
    gpu_group.add_argument('--gpu', type=int, default=0, help='GPU device ID to use (0, 1, 2, ...)')
    gpu_group.add_argument('--use_multi_gpu', action='store_true', help='use multiple GPUs for parallel processing', default=False)
    gpu_group.add_argument('--devices', type=str, default='0,1,2,3', help='device ID list for multi-GPU usage (comma-separated)')
    gpu_group.add_argument('--pin_gpu', type=str_to_bool, default=True, help='whether to pin GPU memory (speeds up memory transfer)')
    gpu_group.add_argument('--test_flop', action='store_true', default=False, help='run FLOPs (floating point operations) test (see utils/tools)')
    gpu_group.add_argument("--local-rank", default=os.getenv('LOCAL_RANK', -1), type=int, help='local rank for distributed training (-1: single GPU, 0+: distributed)')


    return parser


# =====================================
# GPU Manager Class
# =====================================

class GPUManager:
    """Class responsible for GPU configuration and management"""

    def __init__(self, args):
        self.args = args
        self.platform = platform.system()
        self._setup_gpu_config()

    def _setup_gpu_config(self):
        """Initialize GPU configuration"""
        # Check if GPU is available
        self.args.use_gpu = True if torch.cuda.is_available() and self.args.use_gpu else False

        # Check number of available GPUs
        if self.args.use_gpu:
            available_gpus = torch.cuda.device_count()
            if available_gpus == 0:
                print("Warning: CUDA is available but no GPU devices found. Using CPU.")
                self.args.use_gpu = False
            elif self.args.gpu >= available_gpus:
                print(f"Warning: GPU {self.args.gpu} is not available. Using GPU 0 instead.")
                self.args.gpu = 0

        # Memory limit settings for Windows environment
        if self.platform == 'Windows':
            self._setup_windows_memory_limit()

        # Multi-GPU configuration
        if self.args.use_gpu and self.args.use_multi_gpu:
            self._setup_multi_gpu()

        # Distributed training configuration
        if self.args.local_rank != -1:
            self._setup_distributed_training()

    def _setup_windows_memory_limit(self):
        """GPU memory limit settings for Windows environment"""
        torch.cuda.set_per_process_memory_fraction(48 / 61, 0)

    def _setup_multi_gpu(self):
        """Multi-GPU configuration"""
        self.args.devices = self.args.devices.replace(' ', '')
        device_ids = self.args.devices.split(',')
        self.args.device_ids = [int(id_) for id_ in device_ids]
        self.args.gpu = self.args.device_ids[0]

    def _setup_distributed_training(self):
        """Distributed training configuration"""
        torch.cuda.set_device(self.args.local_rank)
        self.args.gpu = self.args.local_rank
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
        self.args.num_gpus = torch.cuda.device_count()
        self.args.batch_size = self.args.batch_size // self.args.num_gpus

    def setup_seed(self, seed):
        """Seed setup for GPU"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def clear_cache(self):
        """Clear GPU cache"""
        torch.cuda.empty_cache()
        gc.collect()

    def get_device_info(self):
        """Get device information"""
        if self.args.use_gpu:
            return f"GPU {self.args.gpu}" if not self.args.use_multi_gpu else f"GPUs {self.args.devices}"
        else:
            return "CPU"


# =====================================
# Model Manager Class
# =====================================

class ModelManager:
    """Class responsible for model-specific processing"""

    def __init__(self, args):
        self.args = args
        self._setup_model_config()
        self.setup_online_method_config()

    def _setup_model_config(self):
        """Initialize model configuration"""
        # Process Ensemble settings
        self._handle_ensemble_config()

        # Process Leakage settings
        self._handle_leakage_config()

        # Generate model ID
        self._generate_model_id()

    def _handle_ensemble_config(self):
        """Process Ensemble settings"""
        # Remove Ensemble from name for models other than TCN and FSNet
        if self.args.model.endswith('_Ensemble') and 'TCN' not in self.args.model and 'FSNet' not in self.args.model:
            self.args.model = self.args.model[:-len('_Ensemble')]
            self.args.ensemble = True
        else:
            self.args.ensemble = False

    def _handle_leakage_config(self):
        """Process Leakage settings"""
        if self.args.model.endswith('_leak'):
            self.args.model = self.args.model[:-len('_leak')]
            self.args.leakage = True

        if self.args.online_method and self.args.online_method.endswith('_leak'):
            self.args.online_method = self.args.online_method[:-len('_leak')]
            self.args.leakage = True


    def _generate_model_id(self):
        """Generate model ID"""
        self.args.model_id = f'{self.args.dataset}_{self.args.seq_len}_{self.args.pred_len}_{self.args.model}'
        if self.args.normalization is not None:
            self.args.model_id += '_' + self.args.normalization
        if self.args.decomposition==1:
            self.args.model_id += '_' + 'dc1'

    def setup_online_method_config(self):
        """Configure online method settings"""
        if not self.args.online_method:
            return

        self.args.patience = min(self.args.patience, 3)
        self.args.train_epochs = min(self.args.train_epochs, 25)
        self.args.save_opt = True

        # FSNet-related settings
        if 'FSNet' in self.args.model and self.args.online_method == 'Online':
            self.args.online_method = 'FSNet'

        if self.args.online_method == 'FSNet' and 'TCN' in self.args.model:
            self.args.model = self.args.model.replace('TCN', 'FSNet')

    def get_seed_offset(self, iteration):
        """Get seed offset according to model"""
        return 2025 + iteration

    def get_model_info(self):
        """Get model information"""
        info = f"Model: {self.args.model}"
        if self.args.ensemble:
            info += " (Ensemble)"
        if self.args.leakage:
            info += " (Leakage)"
        if self.args.normalization:
            info += f" ({self.args.normalization})"
        if self.args.decomposition==1:
            info += f" (dc1)"
        return info


# =====================================
# Result Manager Class
# =====================================

class ResultManager:
    """Class responsible for result processing"""

    def __init__(self, savepath):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        self.json_path = os.path.join(savepath, 'results.json')
        # Initialize with empty dict to allow dynamic phase addition
        self.results = {}
        self.skip_file_processed = False

    def _ensure_phase_exists(self, phase):
        """Create phase if it doesn't exist"""
        if phase not in self.results:
            self.results[phase] = {'mse': [], 'mae': [], 'seed': []}

    def _has_duplicate_seed(self, phase, seed):
        """Check if the same seed already exists in the specified phase"""
        if phase not in self.results:
            return False
        return seed in self.results[phase]['seed']

    def _remove_duplicate_seed(self, phase, seed):
        """Remove results with the same seed in the specified phase"""
        if phase not in self.results:
            return

        try:
            # Get index of seed
            seed_index = self.results[phase]['seed'].index(seed)
            # Remove corresponding mse, mae, seed
            self.results[phase]['mse'].pop(seed_index)
            self.results[phase]['mae'].pop(seed_index)
            self.results[phase]['seed'].pop(seed_index)
        except ValueError:
            # Do nothing if seed is not found
            pass

    def add_result(self, mse, mae, seed=None, phase='test'):
        """Add result (replace if seed is duplicate)

        Args:
            mse: MSE value
            mae: MAE value
            seed: Seed value
            phase: Phase (any string, automatically created if it doesn't exist)
        """
        # Create phase automatically if it doesn't exist
        self._ensure_phase_exists(phase)

        # Remove if the same seed already exists
        if seed is not None and self._has_duplicate_seed(phase, seed):
            self._remove_duplicate_seed(phase, seed)

        self.results[phase]['mse'].append(mse)
        self.results[phase]['mae'].append(mae)
        self.results[phase]['seed'].append(seed)
        self.skip_file_processed = False

    def load_from_skip_file(self, current_seed):
        """
        Load results from skip file and determine whether to skip experiment with specified seed

        Args:
            current_seed (int): Seed value used in current experiment

        Returns:
            bool: True if result for current_seed already exists (skip experiment)
                  False if result doesn't exist (run experiment)

        Note:
            - Found results are automatically added to self.results (avoiding duplicates)
        """
        # If result file doesn't exist, don't skip (run experiment)
        if not os.path.exists(self.json_path):
            return False

        try:
            # Load JSON file and add results for current_seed to self.results
            with open(self.json_path, 'rt', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)

                # Support new data structure (results saved by phase)
                if 'raw_results' in data:
                    raw_results = data['raw_results']
                    found_any_result = False

                    # Process all phases (dynamic support)
                    # e.g., train, valid, test, test1, test2, test3, etc.
                    for phase in raw_results.keys():
                        # Confirm that phase is in dict format and has seed key
                        if isinstance(raw_results[phase], dict) and 'seed' in raw_results[phase]:
                            # Check all seeds in this phase
                            for i, seed in enumerate(raw_results[phase]['seed']):
                                # If result matching current seed is found
                                if seed == current_seed:
                                    # Get corresponding MSE and MAE
                                    mse = float(raw_results[phase]['mse'][i])
                                    mae = float(raw_results[phase]['mae'][i])

                                    # Add result to self.results (add_result method handles duplicate check)
                                    self.add_result(mse, mae, seed, phase)
                                    found_any_result = True

                    # Skip if at least one result is found
                    if found_any_result:
                        self.skip_file_processed = True
                        return True

        except Exception as e:
            # Display warning and run experiment if file loading error occurs
            print(f"Warning: Failed to load skip file: {e}")

        # Run experiment if result for current_seed is not found
        return False

    def get_final_results(self):
        """Get final results (mean and standard deviation)"""
        final_results = {}
        # Process all phases (dynamic support)
        for phase in self.results.keys():
            final_results[phase] = {}
            for metric in ['mse', 'mae']:
                if len(self.results[phase][metric]) > 0:
                    results_array = np.array(self.results[phase][metric])
                    final_results[phase][metric] = [results_array.mean(), results_array.std()]
                else:
                    final_results[phase][metric] = [0.0, 0.0]
        return final_results

    def display_results(self):
        """Display results (including existing results)"""
        # Load and display existing result file if it exists
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'rt', encoding='utf-8', errors='ignore') as f:
                    data = json.load(f)
                    if 'results' in data:
                        final_results = data['results']
                    else:
                        final_results = self.get_final_results()
                print("\n" + "="*50)
                print("FINAL RESULTS (including existing results)")
                print("="*50)
            except Exception as e:
                print(f"Warning: Failed to load existing results for display: {e}")
                final_results = self.get_final_results()
                print("\n" + "="*50)
                print("FINAL RESULTS")
                print("="*50)
        else:
            final_results = self.get_final_results()
            print("\n" + "="*50)
            print("FINAL RESULTS")
            print("="*50)

        # Process all phases (dynamic support)
        for phase in sorted(final_results.keys()):
            print(f"\n{phase.upper()} RESULTS:")
            for metric, values in final_results[phase].items():
                mean_val, std_val = values
                print(f"  {metric.upper()}: {mean_val:.6f} ± {std_val:.6f}")
        print("="*50)
        return final_results

    def _merge_results_without_duplicates(self, existing_results, current_results):
        """Merge existing results with current results avoiding duplicates"""
        merged_results = existing_results.copy()

        for phase in current_results.keys():
            if phase not in merged_results:
                merged_results[phase] = {'mse': [], 'mae': [], 'seed': []}

            # Process each entry in current results
            for i, seed in enumerate(current_results[phase]['seed']):
                if seed is not None:
                    # Check if the same seed exists in existing results
                    if seed in merged_results[phase]['seed']:
                        # Remove entry with same seed from existing results
                        existing_index = merged_results[phase]['seed'].index(seed)
                        merged_results[phase]['mse'].pop(existing_index)
                        merged_results[phase]['mae'].pop(existing_index)
                        merged_results[phase]['seed'].pop(existing_index)

                # Add new result
                merged_results[phase]['mse'].append(current_results[phase]['mse'][i])
                merged_results[phase]['mae'].append(current_results[phase]['mae'][i])
                merged_results[phase]['seed'].append(seed)

        return merged_results

    def save_results(self, experiment_info=None):
        """Save results to file (merge with existing results avoiding duplicates)"""
        try:
            # Load existing result file if it exists
            existing_results = {}
            if os.path.exists(self.json_path):
                try:
                    with open(self.json_path, 'rt', encoding='utf-8', errors='ignore') as f:
                        existing_data = json.load(f)
                        if 'raw_results' in existing_data:
                            existing_results = existing_data['raw_results']
                except Exception as e:
                    print(f"Warning: Failed to load existing results: {e}")
                    existing_results = {}

            # Merge existing results with current results avoiding duplicates
            merged_results = self._merge_results_without_duplicates(existing_results, self.results)

            # Calculate final results
            final_results = {}
            for phase in merged_results.keys():
                final_results[phase] = {}
                for metric in ['mse', 'mae']:
                    if len(merged_results[phase][metric]) > 0:
                        results_array = np.array(merged_results[phase][metric])
                        final_results[phase][metric] = [results_array.mean(), results_array.std()]
                    else:
                        final_results[phase][metric] = [0.0, 0.0]

            save_data = {
                'results': final_results,
                'raw_results': merged_results,
                'timestamp': datetime.datetime.now().isoformat(),
                'experiment_info': experiment_info or {}
            }

            with open(self.json_path, 'w') as f:
                json.dump(save_data, f, indent=2)

            print(f"Results saved to: {self.json_path}")
            return True
        except Exception as e:
            print(f"Warning: Failed to save results: {e}")
            return False

    def get_summary_stats(self):
        """Get statistical summary of results"""
        summary = {}
        # Process all phases (dynamic support)
        for phase in self.results.keys():
            summary[phase] = {}
            for metric in ['mse', 'mae']:
                if len(self.results[phase][metric]) > 0:
                    values_array = np.array(self.results[phase][metric])
                    summary[phase][metric] = {
                        'count': len(self.results[phase][metric]),
                        'mean': float(values_array.mean()),
                        'std': float(values_array.std()),
                        'min': float(values_array.min()),
                        'max': float(values_array.max()),
                        'median': float(np.median(values_array))
                    }
                else:
                    summary[phase][metric] = {
                        'count': 0,
                        'mean': 0.0,
                        'std': 0.0,
                        'min': 0.0,
                        'max': 0.0,
                        'median': 0.0
                    }

        return summary

    def reset(self):
        """Reset results"""
        # Initialize with empty dict to allow dynamic phase addition
        self.results = {}
        self.skip_file_processed = False


def load_checkpoint_for_online_learning(exp, args):
    """Load checkpoint for online learning"""
    if not args.checkpoints:
        print('Checkpoints is not specified')
        return False

    try:
        from util.test_if_model_learned import comprehensive_model_check
        comprehensive_model_check(exp.model)
        print('load checkpoint from', args.load_path)
        exp.load_checkpoint(args.load_path)
        comprehensive_model_check(exp.model)
        print('check if model is learned')
        return True
    except Exception as e:
        print(f"Warning: Failed to load checkpoint: {e}")
        return False


def setup_online_optimizer(exp, args):
    """Configure optimizer for online learning"""
    if hasattr(exp, 'model_optim') and exp.model_optim is not None:
        for j in range(len(exp.model_optim.param_groups)):
            exp.model_optim.param_groups[j]['lr'] = args.online_learning_rate
        print('Online learning rate of model_optim is', exp.model_optim.param_groups[0]['lr'])
        return True
    else:
        print('model_optim not available for this experiment class')
        return False


def run_offline_training_for_online_method(exp, ii, gpu_manager):
    """Execute offline training for online method"""
    print('>>>>>>>offline training for online method : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(ii))
    exp.train()
    gpu_manager.clear_cache()


def run_online_phase(exp, args, result_manager, fix_seed, ii, gpu_manager, phase):
    """Execute each phase of online learning"""
    print(f'>>>>>>>Online {phase} : {ii}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    save_phase = phase+'_online'
    mse, mae = exp.online(phase=phase, savename=save_phase, show_progress=True)
    result_manager.add_result(mse, mae, fix_seed, save_phase)
    gpu_manager.clear_cache()

def run_offline_test(exp, args, result_manager, fix_seed, ii, gpu_manager, phase, timing=''):
    """Execute each phase of offline learning"""
    print(f'>>>>>>>Offline {phase}{timing} : {ii}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    save_phase = phase+'_offline'+timing
    mse, mae = exp.test(phase=phase, savename=save_phase)
    result_manager.add_result(mse, mae, fix_seed, save_phase)
    gpu_manager.clear_cache()


def run_online_learning_experiment(exp, args, result_manager, fix_seed, ii, gpu_manager):
    """Execute online learning experiment"""
    # Load checkpoint
    checkpoint_loaded = load_checkpoint_for_online_learning(exp, args)

    # If offline training is needed
    if args.train:
        run_offline_training_for_online_method(exp, ii, gpu_manager)

    # Configure optimizer for online learning
    setup_online_optimizer(exp, args)

    # Execute each phase of online learning
    if args.online_train:
        run_online_phase(exp, args, result_manager, fix_seed, ii, gpu_manager, 'train')

    if args.online_valid:
        run_online_phase(exp, args, result_manager, fix_seed, ii, gpu_manager, 'valid')

    if args.online_test:
        if args.border_type == 'online':
            run_offline_test(exp, args, result_manager, fix_seed, ii, gpu_manager, 'test', timing='_beforeOnline')
            run_online_phase(exp, args, result_manager, fix_seed, ii, gpu_manager, 'test')
            run_offline_test(exp, args, result_manager, fix_seed, ii, gpu_manager, 'test', timing='_afterOnline')

        elif args.border_type == 'online3test':
            if args.online_method in ['Chronos']: # If model is not updated
                run_online_phase(exp, args, result_manager, fix_seed, ii, gpu_manager, 'test1')
                run_online_phase(exp, args, result_manager, fix_seed, ii, gpu_manager, 'test2')
                run_online_phase(exp, args, result_manager, fix_seed, ii, gpu_manager, 'test3')
            else:
                # Test before online learning
                run_offline_test(exp, args, result_manager, fix_seed, ii, gpu_manager, 'test1', timing='_beforeOnlinetest1')
                run_offline_test(exp, args, result_manager, fix_seed, ii, gpu_manager, 'test2', timing='_beforeOnlinetest1')
                run_offline_test(exp, args, result_manager, fix_seed, ii, gpu_manager, 'test3', timing='_beforeOnlinetest1')

                run_online_phase(exp, args, result_manager, fix_seed, ii, gpu_manager, 'test1')
                run_offline_test(exp, args, result_manager, fix_seed, ii, gpu_manager, 'test1', timing='_afterOnlinetest1')
                run_offline_test(exp, args, result_manager, fix_seed, ii, gpu_manager, 'test2', timing='_afterOnlinetest1')
                run_offline_test(exp, args, result_manager, fix_seed, ii, gpu_manager, 'test3', timing='_afterOnlinetest1')

                run_online_phase(exp, args, result_manager, fix_seed, ii, gpu_manager, 'test2')
                run_offline_test(exp, args, result_manager, fix_seed, ii, gpu_manager, 'test1', timing='_afterOnlinetest2')
                run_offline_test(exp, args, result_manager, fix_seed, ii, gpu_manager, 'test2', timing='_afterOnlinetest2')
                run_offline_test(exp, args, result_manager, fix_seed, ii, gpu_manager, 'test3', timing='_afterOnlinetest2')

                run_online_phase(exp, args, result_manager, fix_seed, ii, gpu_manager, 'test3')
                run_offline_test(exp, args, result_manager, fix_seed, ii, gpu_manager, 'test1', timing='_afterOnlinetest3')
                run_offline_test(exp, args, result_manager, fix_seed, ii, gpu_manager, 'test2', timing='_afterOnlinetest3')
                run_offline_test(exp, args, result_manager, fix_seed, ii, gpu_manager, 'test3', timing='_afterOnlinetest3')


def run_offline_learning_experiment(exp, args, result_manager, fix_seed, ii, gpu_manager):
    """Execute offline learning experiment"""
    if args.train:
        print('>>>>>>>offline learning training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(ii))
        exp.train()
        gpu_manager.clear_cache()

    if args.valid:
        run_offline_test(exp, args, result_manager, fix_seed, ii, gpu_manager, 'valid')

    if args.test:
        if args.border_type == 'online':
            run_offline_test(exp, args, result_manager, fix_seed, ii, gpu_manager, 'test')
        elif args.border_type == 'online3test':
            run_offline_test(exp, args, result_manager, fix_seed, ii, gpu_manager, 'test1')
            run_offline_test(exp, args, result_manager, fix_seed, ii, gpu_manager, 'test2')
            run_offline_test(exp, args, result_manager, fix_seed, ii, gpu_manager, 'test3')


def run_single_experiment_iteration(args, result_manager, model_manager, gpu_manager, ii):
    """Execute a single experiment iteration"""
    # Configure paths
    args.ii = ii
    args.savepath_itr = os.path.join(args.savepath, str(ii))

    if args.checkpoints:
        args.checkpoints_itr = os.path.join(args.checkpoints, str(ii))
        args.load_path = os.path.join(args.checkpoints_itr, 'checkpoints', 'checkpoint.pth')
        print('Checkpoints in', args.load_path)

    # Configure seed
    fix_seed = model_manager.get_seed_offset(ii)
    gpu_manager.setup_seed(fix_seed)
    print('Seed:', fix_seed)

    # Process skip file (reuse existing results)
    if args.skip:
        if result_manager.load_from_skip_file(fix_seed):
            print('Skip this experiment with seed {} (result already exists)'.format(fix_seed))
            return

    # Determine experiment class
    if args.online_method:
        Exp = getattr(exps, 'Exp_' + args.online_method)
    else:
        Exp = Exp_Main

    # Override hyperparameters
    if args.override_hyper and args.model in settings.hyperparams:
        for k, v in settings.get_hyperparams(args.dataset, args.model, args).items():
            args.__setattr__(k, v)

    # Create experiment object
    exp = Exp(args)

    print('Args in experiment:')
    print(args)
    print(f'Using device: {gpu_manager.get_device_info()}')
    print(f'{model_manager.get_model_info()}')

    # Execute experiment
    if not args.online_method:
        run_offline_learning_experiment(exp, args, result_manager, fix_seed, ii, gpu_manager)
    else:
        run_online_learning_experiment(exp, args, result_manager, fix_seed, ii, gpu_manager)


def setup_experiment_args(args):
    """Configure experiment arguments"""
    # Configure input/output dimensions and paths for each dataset
    args.enc_in, args.c_out = data_settings[args.dataset][args.features]
    args.dec_in = args.enc_in
    args.data_path = data_settings[args.dataset]['data']
    args.data = args.data_path[:5] if args.data_path.startswith('ETT') else 'custom'
    args.timeenc = 2  # Time encoding method


def main():
    """
    Main function - controls overall experimental flow
    - Parse arguments, initialize various managers
    - Manage data, models, paths, and results
    - Experiment loop (seed, training, validation, testing, prediction)
    - Aggregate and display results
    """
    # Display current time (for logging)
    cur_sec = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(cur_sec)

    # Use structured argument parser
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Initialize various manager classes
    gpu_manager = GPUManager(args)
    model_manager = ModelManager(args)
    result_manager = ResultManager(args.savepath)

    # Configure experiment arguments
    setup_experiment_args(args)

    # Repeat experiments (args.itr times)
    for ii in range(args.itr):
        run_single_experiment_iteration(args, result_manager, model_manager, gpu_manager, ii)
        # Save results after each iteration (including when skipped)
        result_manager.save_results()

    # Display and save final results
    result_manager.display_results()
    result_manager.save_results()


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
リファクタリング版の実験実行スクリプト
引数定義の改善版 - カテゴリ別に整理して可読性を向上
"""

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
from exp.exp_solid import Exp_SOLID
from settings import data_settings


def str_to_bool(value):
    """文字列をブール値に変換するヘルパー関数"""
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def setup_argument_parser():
    """構造化された引数解析器を作成"""
    parser = argparse.ArgumentParser(description='Online Time Series Forecasting')

    # =====================================
    # 基本設定
    # =====================================
    basic_group = parser.add_argument_group('Basic Configuration')
    basic_group.add_argument('--train_only', action='store_true', default=False,
                            help='perform training on full input dataset without validation and testing')
    basic_group.add_argument('--wo_test', action='store_true', default=False, help='only valid, not test')
    basic_group.add_argument('--wo_valid', action='store_true', default=False, help='only test')
    basic_group.add_argument('--only_test', action='store_true', default=False)
    basic_group.add_argument('--do_valid', action='store_true', default=False)
    basic_group.add_argument('--model', type=str, required=True, default='PatchTST')
    basic_group.add_argument('--override_hyper', action='store_true', default=True, help='Override hyperparams by setting.py')
    basic_group.add_argument('--compile', action='store_true', default=False, help='Compile the model by Pytorch 2.0')
    basic_group.add_argument('--reduce_bs', type=str_to_bool, default=False,
                            help='Override batch_size in hyperparams by setting.py')
    basic_group.add_argument('--normalization', type=str, default=None)
    basic_group.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    basic_group.add_argument('--tag', type=str, default='')

    # =====================================
    # オンライン学習設定
    # =====================================
    online_group = parser.add_argument_group('Online Learning')
    online_group.add_argument('--online_method', type=str, default=None)
    online_group.add_argument('--skip', type=str, default=None)
    online_group.add_argument('--online_learning_rate', type=float, default=None)
    online_group.add_argument('--val_online_lr', action='store_true', default=True)
    online_group.add_argument('--diff_online_lr', action='store_true', default=False)
    online_group.add_argument('--save_opt', action='store_true', default=True)
    online_group.add_argument('--leakage', action='store_true', default=False)
    online_group.add_argument('--debug', action='store_true', default=False)
    online_group.add_argument('--pretrain', action='store_true', default=False)
    online_group.add_argument('--freeze', action='store_true', default=False)

    # =====================================
    # Proceed手法設定
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
    # OneNet手法設定
    # =====================================
    onenet_group = parser.add_argument_group('OneNet Method')
    onenet_group.add_argument('--learning_rate_w', type=float, default=0.001, help='optimizer learning rate')
    onenet_group.add_argument('--learning_rate_bias', type=float, default=0.001, help='optimizer learning rate')

    # =====================================
    # データ設定
    # =====================================
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--border_type', type=str, default='online', help='set any other value for traditional data splits')
    data_group.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    data_group.add_argument('--dataset', type=str, default='ETTh1', help='data file')
    data_group.add_argument('--features', type=str, default='M',
                           help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    data_group.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    data_group.add_argument('--freq', type=str, default='h',
                           help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    data_group.add_argument('--wrap_data_class', type=list, default=[])
    data_group.add_argument('--pin_gpu', type=str_to_bool, default=True)
    data_group.add_argument('--use_time', action='store_true', default=False, help='use time features or not')

    # =====================================
    # 予測タスク設定
    # =====================================
    forecasting_group = parser.add_argument_group('Forecasting Task')
    forecasting_group.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    forecasting_group.add_argument('--label_len', type=int, default=48, help='start token length')
    forecasting_group.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    forecasting_group.add_argument('--individual', action='store_true', default=False,
                                  help='DLinear: a linear layer for each variate(channel) individually')

    # =====================================
    # PatchTST設定
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
    # Transformer系設定
    # =====================================
    transformer_group = parser.add_argument_group('Transformer Configuration')
    transformer_group.add_argument('--embed_type', type=int, default=0,
                                  help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    transformer_group.add_argument('--d_model', type=int, default=512, help='dimension of model')
    transformer_group.add_argument('--n_heads', type=int, default=8, help='num of heads')
    transformer_group.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    transformer_group.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    transformer_group.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    transformer_group.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    transformer_group.add_argument('--factor', type=int, default=3, help='attn factor')
    transformer_group.add_argument('--distil', action='store_false',
                                  help='whether to use distilling in encoder, using this argument means not using distilling',
                                  default=True)
    transformer_group.add_argument('--dropout', type=float, default=0.05, help='dropout')
    transformer_group.add_argument('--embed', type=str, default='timeF',
                                  help='time features encoding, options:[timeF, fixed, learned]')
    transformer_group.add_argument('--activation', type=str, default='gelu', help='activation')
    transformer_group.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    transformer_group.add_argument('--output_enc', action='store_true', help='whether to output embedding from encoder')
    transformer_group.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # =====================================
    # Crossformer設定
    # =====================================
    crossformer_group = parser.add_argument_group('Crossformer Configuration')
    crossformer_group.add_argument('--seg_len', type=int, default=24, help='segment length (L_seg)')
    crossformer_group.add_argument('--win_size', type=int, default=2, help='window size for segment merge')
    crossformer_group.add_argument('--num_routers', type=int, default=10, help='num of routers in Cross-Dimension Stage of TSA (c)')

    # =====================================
    # その他モデル設定
    # =====================================
    other_models_group = parser.add_argument_group('Other Model Configuration')
    other_models_group.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    other_models_group.add_argument('--subgraph_size', type=int, default=20, help='k')
    other_models_group.add_argument('--in_dim', type=int, default=1)
    other_models_group.add_argument('--gpt_layers', type=int, default=6)
    other_models_group.add_argument('--tmax', type=int, default=10)
    other_models_group.add_argument('--patch_size', type=int, default=16)

    # =====================================
    # 最適化設定
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
    # GPU設定
    # =====================================
    gpu_group = parser.add_argument_group('GPU Configuration')
    gpu_group.add_argument('--use_gpu', type=str_to_bool, default=True, help='use gpu')
    gpu_group.add_argument('--gpu', type=int, default=0, help='gpu')
    gpu_group.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    gpu_group.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    gpu_group.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
    gpu_group.add_argument("--local-rank", default=os.getenv('LOCAL_RANK', -1), type=int)

    # =====================================
    # SOLID設定
    # =====================================
    solid_group = parser.add_argument_group('SOLID Configuration')
    solid_group.add_argument('--test_train_num', type=int, default=500)
    solid_group.add_argument('--selected_data_num', type=int, default=5)
    solid_group.add_argument('--lambda_period', type=float, default=0.1)
    solid_group.add_argument('--whole_model', action='store_true')
    solid_group.add_argument('--continual', action='store_true')

    return parser


# =====================================
# GPU管理クラス
# =====================================

class GPUManager:
    """GPU設定・管理を担当するクラス"""

    def __init__(self, args):
        self.args = args
        self.platform = platform.system()
        self._setup_gpu_config()

    def _setup_gpu_config(self):
        """GPU設定の初期化"""
        # GPU使用可能かチェック
        self.args.use_gpu = True if torch.cuda.is_available() and self.args.use_gpu else False

        # Windows環境でのメモリ制限設定
        if self.platform == 'Windows':
            self._setup_windows_memory_limit()

        # マルチGPU設定
        if self.args.use_gpu and self.args.use_multi_gpu:
            self._setup_multi_gpu()

        # 分散学習設定
        if self.args.local_rank != -1:
            self._setup_distributed_training()

    def _setup_windows_memory_limit(self):
        """Windows環境でのGPUメモリ制限設定"""
        torch.cuda.set_per_process_memory_fraction(48 / 61, 0)

    def _setup_multi_gpu(self):
        """マルチGPU設定"""
        self.args.devices = self.args.devices.replace(' ', '')
        device_ids = self.args.devices.split(',')
        self.args.device_ids = [int(id_) for id_ in device_ids]
        self.args.gpu = self.args.device_ids[0]

    def _setup_distributed_training(self):
        """分散学習設定"""
        torch.cuda.set_device(self.args.local_rank)
        self.args.gpu = self.args.local_rank
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
        self.args.num_gpus = torch.cuda.device_count()
        self.args.batch_size = self.args.batch_size // self.args.num_gpus

    def setup_seed(self, seed):
        """GPU対応のシード設定"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def clear_cache(self):
        """GPUキャッシュのクリア"""
        torch.cuda.empty_cache()

    def get_device_info(self):
        """デバイス情報を取得"""
        if self.args.use_gpu:
            return f"GPU {self.args.gpu}" if not self.args.use_multi_gpu else f"GPUs {self.args.devices}"
        else:
            return "CPU"


# =====================================
# パス管理クラス
# =====================================

class PathManager:
    """パス生成・管理を担当するクラス"""

    def __init__(self, args):
        self.args = args
        self.platform = platform.system()
        self._setup_base_paths()

    def _setup_base_paths(self):
        """基本パスの設定"""
        if self.platform != 'Windows':
            self.base_path = './'
            self.checkpoints_base = './checkpoints/'
            self.results_base = './results/'
        else:
            self.base_path = 'D:/data/'
            self.checkpoints_base = 'D:/checkpoints/'
            self.results_base = './results/'

        # Windowsの場合、checkpointsパスを上書き
        if self.platform == 'Windows' and self.args.checkpoints:
            self.args.checkpoints = self.checkpoints_base

    def get_data_path(self):
        """データパスを取得"""
        return data_settings[self.args.dataset]['data']

    def get_checkpoint_path(self, setting):
        """チェックポイントパスを生成"""
        return os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')

    def get_pretrain_paths(self, pretrain_setting):
        """プリトレーニング用パスを生成"""
        pred_path = os.path.join(self.results_base, pretrain_setting, 'real_prediction.npy')

        if self.platform == 'Windows':
            load_path = os.path.join(self.checkpoints_base, pretrain_setting, 'checkpoint.pth')
        else:
            load_path = os.path.join('./checkpoints/', pretrain_setting, 'checkpoint.pth')

        return pred_path, load_path

    def get_fsnet_path(self, fsnet_name, iteration):
        """FSNet用パスを生成"""
        return f'./checkpoints/{self.args.dataset}_60_{self.args.pred_len}_{fsnet_name}_' \
               f'online_ftM_sl60_ll48_pl{self.args.pred_len}_lr{settings.pretrain_lr_online_dict[fsnet_name][self.args.dataset]}' \
               f'_uniFalse_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_test_{iteration}/checkpoint.pth'

    def get_setting_name(self, flag, iteration):
        """実験設定名を生成"""
        return '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            self.args.model_id,
            flag,
            self.args.features,
            self.args.seq_len,
            self.args.label_len,
            self.args.pred_len,
            self.args.learning_rate,
            self.args.d_model,
            self.args.n_heads,
            self.args.e_layers,
            self.args.d_layers,
            self.args.d_ff,
            self.args.factor,
            self.args.embed,
            self.args.distil,
            self.args.des, iteration)

    def get_pretrain_setting_name(self, pretrain_lr, iteration):
        """プリトレーニング設定名を生成"""
        return '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            self.args.model_id,
            self.args.border_type if self.args.border_type else self.args.data,
            self.args.features,
            self.args.seq_len,
            self.args.label_len,
            self.args.pred_len,
            pretrain_lr,
            self.args.d_model,
            self.args.n_heads,
            self.args.e_layers,
            self.args.d_layers,
            self.args.d_ff,
            self.args.factor,
            self.args.embed,
            self.args.distil,
            self.args.des, iteration)

    def generate_flag(self):
        """
        flagを生成（実験設定に基づいて設定名の一部を生成）

        Returns:
            str: 生成されたflag
        """
        if self.args.online_method:
            flag = self.args.online_method.lower()
            if not self.args.border_type:
                if self.args.online_method == 'Online':
                    flag = self.args.data
                    self.args.checkpoints = ""
                else:
                    flag = self.args.data + '_' + flag

            if flag == 'fsnet':
                flag = 'online'

            if 'proceed' in flag:
                if not self.args.freeze:
                    flag += "_fulltune"
                if not self.args.pretrain:
                    flag += "_new"
                flag += f"_{self.args.lradj}"
                flag += f'_{self.args.tune_mode}_btl{self.args.bottleneck_dim}_ema{self.args.ema}'
                if self.args.concept_dim:
                    flag += f'_mid{self.args.concept_dim}'
                if not self.args.individual_generator:
                    flag += '_share'
                if self.args.share_encoder:
                    flag += '_share_enc'
                if self.args.wo_clip:
                    flag += '_noclip'
        else:
            flag = self.args.border_type if self.args.border_type else self.args.data

        return flag


# =====================================
# モデル管理クラス
# =====================================

class ModelManager:
    """モデル固有の処理を担当するクラス"""

    def __init__(self, args):
        self.args = args
        self._setup_model_config()

    def _setup_model_config(self):
        """モデル設定の初期化"""
        # Ensemble設定の処理
        self._handle_ensemble_config()

        # Leakage設定の処理
        self._handle_leakage_config()

        # モデル固有の設定
        self._setup_model_specific_config()

        # モデルIDの生成
        self._generate_model_id()

    def _handle_ensemble_config(self):
        """Ensemble設定の処理"""
        if self.args.model.endswith('_Ensemble') and 'TCN' not in self.args.model and 'FSNet' not in self.args.model:
            self.args.model = self.args.model[:-len('_Ensemble')]
            self.args.ensemble = True
        else:
            self.args.ensemble = False

    def _handle_leakage_config(self):
        """Leakage設定の処理"""
        if self.args.model.endswith('_leak'):
            self.args.model = self.args.model[:-len('_leak')]
            self.args.leakage = True

        if self.args.online_method and self.args.online_method.endswith('_leak'):
            self.args.online_method = self.args.online_method[:-len('_leak')]
            self.args.leakage = True

    def _setup_model_specific_config(self):
        """モデル固有の設定"""
        # GPT4TS固有の設定
        if self.args.model.startswith('GPT4TS'):
            self._setup_gpt4ts_config()

        # MTGNN固有の設定
        if self.args.model in ['MTGNN']:
            self._setup_mtgnn_config()

        # 特殊な最適化設定
        if self.args.model in settings.need_x_mark:
            self.args.optim = 'AdamW'
            self.args.patience = 3

        # 未使用パラメータの設定
        self.args.find_unused_parameters = self.args.model in ['MTGNN']

    def _setup_gpt4ts_config(self):
        """GPT4TS固有の設定"""
        if not self.args.online_method and not self.args.do_predict:
            self.args.data += '_CI'
        else:
            if self.args.dataset == 'ECL':
                self.args.batch_size = min(self.args.batch_size, 3)
            elif self.args.dataset == 'Traffic':
                self.args.batch_size = 1

    def _setup_mtgnn_config(self):
        """MTGNN固有の設定"""
        if 'feat_dim' in data_settings[self.args.dataset]:
            self.args.in_dim = data_settings[self.args.dataset]['feat_dim']
            self.args.enc_in = int(self.args.enc_in / self.args.in_dim)
            if self.args.features == 'M':
                self.args.c_out = int(self.args.c_out / self.args.in_dim)

    def _generate_model_id(self):
        """モデルIDの生成"""
        self.args.model_id = f'{self.args.dataset}_{self.args.seq_len}_{self.args.pred_len}_{self.args.model}'
        if self.args.normalization is not None:
            self.args.model_id += '_' + self.args.normalization

    def setup_online_method_config(self):
        """オンライン手法の設定"""
        if not self.args.online_method:
            return

        self.args.train_epochs = min(self.args.train_epochs, 25)
        self.args.save_opt = True

        # FSNet関連の設定
        if 'FSNet' in self.args.model and self.args.online_method == 'Online':
            self.args.online_method = 'FSNet'

        if self.args.online_method == 'FSNet' and 'TCN' in self.args.model:
            self.args.model = self.args.model.replace('TCN', 'FSNet')

        # オンライン手法固有の設定
        if self.args.online_method == 'Online':
            self.args.pretrain = True
            self.args.only_test = True

        if 'FSNet' in self.args.model:
            self.args.pretrain = False
        elif self.args.online_method.lower() in settings.peft_methods:
            self.args.pretrain = True
            self.args.freeze = True

        # SOLID固有の設定
        if self.args.online_method == 'SOLID':
            self.args.pretrain = True
            self.args.only_test = True
            self.args.online_method = 'Online'
            if not self.args.whole_model:
                self.args.freeze = True

    def get_seed_offset(self, iteration):
        """モデルに応じたシードオフセットを取得"""
        if self.args.border_type:
            if self.args.model in ['PatchTST', 'iTransformer']:
                return 2021 + iteration
            else:
                return 2023 + iteration
        else:
            return 2023 + iteration if self.args.model == 'iTransformer' else 2021 + iteration

    def get_pretrain_lr(self, dataset):
        """モデルに応じたプリトレーニング学習率を取得"""
        model_key = self.args.model + ("_RevIN" if self.args.normalization else "")

        if self.args.online_method:
            pretrain_lr = settings.pretrain_lr_online_dict[model_key][dataset]
        else:
            pretrain_lr = settings.pretrain_lr_dict[self.args.model][dataset]

        # iTransformer + Weather の特殊ケース
        if not self.args.border_type and self.args.model == 'iTransformer' and dataset == 'Weather':
            pretrain_lr = 0.0001

        return pretrain_lr

    def should_drop_last_patchtst(self):
        """PatchTSTでドロップアウトすべきかチェック"""
        return self.args.border_type != 'online' and self.args.model == 'PatchTST'

    def get_model_info(self):
        """モデル情報を取得"""
        info = f"Model: {self.args.model}"
        if self.args.ensemble:
            info += " (Ensemble)"
        if self.args.leakage:
            info += " (Leakage)"
        if self.args.normalization:
            info += f" ({self.args.normalization})"
        return info


# =====================================
# 結果管理クラス
# =====================================

class ResultManager:
    """結果処理を担当するクラス"""

    def __init__(self):
        self.results = {'mse': [], 'mae': []}
        self.skip_file_processed = False

    def add_result(self, mse, mae):
        """結果を追加"""
        self.results['mse'].append(mse)
        self.results['mae'].append(mae)

    def load_from_skip_file(self, skip_file_path):
        """スキップファイルから結果を読み込み"""
        if not skip_file_path or not os.path.exists(skip_file_path):
            return False

        try:
            with open(skip_file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                for line in f.readlines():
                    if line.startswith('mse:'):
                        splits = line.split(',')
                        mse, mae = splits[0].split(':')[1], splits[1].split(':')[1]
                        self.results['mse'].append(float(mse))
                        self.results['mae'].append(float(mae))
                        self.skip_file_processed = True
                        return True
        except Exception as e:
            print(f"Warning: Failed to load skip file: {e}")

        return False

    def has_results(self):
        """結果が存在するかチェック"""
        return len(self.results['mse']) > 0

    def get_final_results(self):
        """最終結果を取得（平均と標準偏差）"""
        final_results = {}
        for k in self.results.keys():
            if len(self.results[k]) > 0:
                results_array = np.array(self.results[k])
                final_results[k] = [results_array.mean(), results_array.std()]
            else:
                final_results[k] = [0.0, 0.0]
        return final_results

    def display_results(self):
        """結果を表示"""
        final_results = self.get_final_results()
        print("\n" + "="*50)
        print("FINAL RESULTS")
        print("="*50)
        for metric, values in final_results.items():
            mean_val, std_val = values
            print(f"{metric.upper()}: {mean_val:.6f} ± {std_val:.6f}")
        print("="*50)
        return final_results

    def save_results(self, file_path, experiment_info=None):
        """結果をファイルに保存"""
        try:
            final_results = self.get_final_results()
            save_data = {
                'results': final_results,
                'raw_results': self.results,
                'timestamp': datetime.datetime.now().isoformat(),
                'experiment_info': experiment_info or {}
            }

            # ディレクトリが存在しない場合は作成
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, 'w') as f:
                json.dump(save_data, f, indent=2)

            print(f"Results saved to: {file_path}")
            return True
        except Exception as e:
            print(f"Warning: Failed to save results: {e}")
            return False

    def get_summary_stats(self):
        """結果の統計サマリーを取得"""
        if not self.has_results():
            return {}

        summary = {}
        for metric, values in self.results.items():
            if len(values) > 0:
                values_array = np.array(values)
                summary[metric] = {
                    'count': len(values),
                    'mean': float(values_array.mean()),
                    'std': float(values_array.std()),
                    'min': float(values_array.min()),
                    'max': float(values_array.max()),
                    'median': float(np.median(values_array))
                }

        return summary

    def reset(self):
        """結果をリセット"""
        self.results = {'mse': [], 'mae': []}
        self.skip_file_processed = False


def main():
    """メイン関数 - 元のrun.pyと同じ処理を実行"""
    # 現在時刻を表示
    cur_sec = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(cur_sec)

    # 構造化された引数解析器を使用
    parser = setup_argument_parser()
    args = parser.parse_args()

    # GPU管理クラスの初期化
    gpu_manager = GPUManager(args)

    # モデル管理クラスの初期化
    model_manager = ModelManager(args)

    # 結果管理クラスの初期化
    result_manager = ResultManager()

    args.enc_in, args.c_out = data_settings[args.dataset][args.features]
    args.data_path = data_settings[args.dataset]['data']
    args.data = args.data_path[:5] if args.data_path.startswith('ETT') else 'custom'
    args.dec_in = args.enc_in
    args.timeenc = 2

    if args.tag and args.tag[0] != '_':
        args.tag = '_' + args.tag

    if hasattr(args, 'border_type'):
        settings.get_borders(args)


    if args.border_type == 'online':
        args.patience = min(args.patience, 3)

    # モデル管理クラスを使用してオンライン手法設定
    model_manager.setup_online_method_config()

    Exp = Exp_Main
    if args.online_method:
        Exp = getattr(exps, 'Exp_' + args.online_method)

    if args.override_hyper and args.model in settings.hyperparams:
        for k, v in settings.get_hyperparams(args.dataset, args.model, args, args.reduce_bs).items():
            args.__setattr__(k, v)

    # パス管理クラスの初期化
    path_manager = PathManager(args)

    # パス管理クラスを使用してflagを生成
    flag = path_manager.generate_flag()

    print('Args in experiment:')
    print(args)
    print(f'Using device: {gpu_manager.get_device_info()}')
    print(f'{model_manager.get_model_info()}')

    # 実験実行部分（元のrun.pyと同じ）
    train_data, train_loader, vali_data, vali_loader = None, None, None, None
    test_data, test_loader = None, None

    for ii in range(args.itr):
        # スキップファイルの処理
        if ii == 0 and args.skip:
            if args.wo_test:
                continue
            if result_manager.load_from_skip_file(args.skip):
                if result_manager.has_results():
                    continue

        # モデル管理クラスを使用してシードオフセット取得
        fix_seed = model_manager.get_seed_offset(ii)

        # GPU管理クラスを使用してシード設定
        gpu_manager.setup_seed(fix_seed)
        print('Seed:', fix_seed)

        setting = path_manager.get_setting_name(flag, ii)

        if args.pretrain:
            # モデル管理クラスを使用してプリトレーニング学習率取得
            pretrain_lr = model_manager.get_pretrain_lr(args.dataset)
			# パス管理クラスを使用してパスを生成
            pretrain_setting = path_manager.get_pretrain_setting_name(pretrain_lr, ii)
            args.pred_path, args.load_path = path_manager.get_pretrain_paths(pretrain_setting)
			if args.online_method == 'OneNet':
				fsnet_name = "FSNet_RevIN"
				args.fsnet_path = path_manager.get_fsnet_path(fsnet_name, ii)

        exp = Exp(args)  # set experiments

        if train_data is None:
            train_data, train_loader = exp._get_data('train')
        if not hasattr(args, 'borders'):
            args.borders = train_data.borders
            # モデル管理クラスを使用してドロップアウト判定
            if model_manager.should_drop_last_patchtst():
                settings.drop_last_PatchTST(args) # SOLID dropout the last when data split = 7:2:1
        exp.wrap_data_kwargs['borders'] = args.borders

        # パス管理クラスを使用してチェックポイントパスを生成
        checkpoint_path = path_manager.get_checkpoint_path(setting)
        if args.online_method not in ['Online', 'SOLID', 'ER', 'DERpp']:
            print('Checkpoints in', checkpoint_path)
			# trainingをしない場合はcheckpointを読み込む
            if (args.only_test or args.do_valid) and os.path.exists(checkpoint_path):
                print('Loading', checkpoint_path)
                exp.load_checkpoint(checkpoint_path)
                print('Learning rate of model_optim is', exp.model_optim.param_groups[0]['lr'])
			# trainingをする場合
            else:
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                _, train_data, train_loader, vali_data, vali_loader = exp.train(setting, train_data, train_loader,
                                                                                vali_data, vali_loader)
                # GPU管理クラスを使用してキャッシュクリア
                gpu_manager.clear_cache()

		# オンライン学習率の調整(SOLIDは除く)
        if args.online_learning_rate is not None and not isinstance(exp, Exp_SOLID):
            for j in range(len(exp.model_optim.param_groups)):
                exp.model_optim.param_groups[j]['lr'] = args.online_learning_rate
            print('Adjust learning rate of model_optim to', exp.model_optim.param_groups[0]['lr'])

        if args.do_valid and args.online_method and args.local_rank <= 0:
            assert isinstance(exp, Exp_Online)
            mse, mae = exp.online(online_data=vali_data if isinstance(vali_data, Dataset_Recent) else None,
                                  phase='val', show_progress=True)[:2]
            print('Best Valid MSE:', mse)
            result_manager.add_result(mse, mae)
            continue

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            gpu_manager.setup_seed(fix_seed)
            mse, mae = exp.predict(checkpoint_path, setting, True)[:2]
            result_manager.add_result(mse, mae)
        elif not args.wo_test and not args.train_only and args.local_rank <= 0:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            if isinstance(exp, Exp_Online):
                gpu_manager.setup_seed(fix_seed)
                if not isinstance(exp, Exp_SOLID) and not args.wo_valid:
                    vali_data = None
                    gpu_manager.clear_cache()
                    gc.collect()
                    exp.update_valid()
                mse, mae, test_data = exp.online(test_data)
            else:
                mse, mae, test_data, test_loader = exp.test(setting, test_data, test_loader)
            result_manager.add_result(mse, mae)
        # GPU管理クラスを使用してキャッシュクリア
        gpu_manager.clear_cache()

    # 結果管理クラスを使用して結果を表示
    result_manager.display_results()


if __name__ == '__main__':
    main()
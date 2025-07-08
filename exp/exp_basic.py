"""
実験の基本クラス
時系列予測実験の共通機能を提供する基底クラス
"""

import os
import warnings
import numpy as np
import typing
import torch
from torch import optim
import torch.nn as nn
from collections import OrderedDict


from data_provider.data_factory import data_provider
from util.tools import remove_state_key_prefix


class Exp_Basic(object):
    """実験の基本クラス - 時系列予測実験の共通機能を提供"""

    def __init__(self, args):
        """
        初期化

        Args:
            args: 実験設定の引数オブジェクト
        """
        self.args = args
        self.label_position = 1  # ラベルの位置（バッチ内のインデックス）
        self.device = self._acquire_device()  # デバイス（GPU/CPU）の取得
        self.wrap_data_kwargs = {}  # データラッパーの追加引数
        self.model_optim = None  # モデルの最適化器
        model = self._build_model()  # モデルの構築
        if model is not None:
            self.model = model.to(self.device)  # モデルをデバイスに移動
            self.model_optim = self._select_optimizer()  # 最適化器の選択

    def _acquire_device(self):
        """
        デバイス（GPU/CPU）を取得

        Returns:
            torch.device: 使用するデバイス
        """
        if self.args.use_gpu:
            # GPU使用時
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            # CPU使用時
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self, model=None, framework_class=None):
        """
        モデルを構築（サブクラスで実装）

        Args:
            model: 既存のモデル（オプション）
            framework_class: フレームワーククラス（オプション）

        Raises:
            NotImplementedError: サブクラスで実装する必要がある
        """
        raise NotImplementedError

    def _get_data(self, flag, **kwargs):
        """
        データセットとデータローダーを取得

        Args:
            flag: データフラグ（'train', 'val', 'test'など）
            **kwargs: 追加の引数

        Returns:
            tuple: (データセット, データローダー)
        """
        data_set, data_loader = data_provider(args=self.args, flag=flag, device=self.device, wrap_class=self.args.wrap_data_class, **self.wrap_data_kwargs, **kwargs)
        return data_set, data_loader

    def _select_optimizer(self, filter_frozen=True, return_self=True, model=None):
        """
        最適化器を選択

        Args:
            filter_frozen: 凍結パラメータをフィルタするかどうか
            return_self: 自身の最適化器を返すかどうか
            model: 対象モデル（Noneの場合はself.model）

        Returns:
            torch.optim.Optimizer: 選択された最適化器
        """
        if return_self and self.model_optim is not None:
            return self.model_optim
        else:
            # 新しい最適化器をインスタンス化
            params = self.model.parameters() if model is None else model.parameters()
            if filter_frozen:
                params = filter(lambda p: p.requires_grad, params)  # 勾配計算が必要なパラメータのみ
            if not hasattr(self.args, 'optim'):
                self.args.optim = 'Adam'  # デフォルトはAdam
            model_optim = getattr(optim, self.args.optim)(params, lr=self.args.learning_rate)
            if return_self:
                self.model_optim = model_optim
            return model_optim

    def _select_criterion(self):
        """
        損失関数を選択

        Returns:
            nn.Module: 選択された損失関数（デフォルトはMSE）
        """
        criterion = nn.MSELoss()
        return criterion

    def _process_batch(self, batch):
        """
        バッチデータを前処理

        Args:
            batch: 入力バッチデータ

        Returns:
            処理済みのバッチデータ
        """
        return batch

    def forward(self, batch):
        """
        順伝播処理

        Args:
            batch: 入力バッチデータ

        Returns:
            モデルの出力
        """
        if not self.args.pin_gpu:
            # GPU固定が無効な場合、テンソルをデバイスに移動
            batch = [batch[i].to(self.device) if isinstance(batch[i], torch.Tensor) and i != self.label_position
                     else batch[i] for i in range(len(batch))]
        inp = self._process_batch(batch)  # バッチの前処理
        if self.args.use_amp:
            # 自動混合精度を使用
            with torch.cuda.amp.autocast():
                outputs = self.model(*inp)
        else:
            outputs = self.model(*inp)
        return outputs

    def train_loss(self, criterion, batch, outputs):
        """
        訓練時の損失計算

        Args:
            criterion: 損失関数
            batch: 入力バッチデータ
            outputs: モデルの出力

        Returns:
            torch.Tensor: 計算された損失
        """
        batch_y = batch[1]  # ラベルデータ
        if not self.args.pin_gpu:
            batch_y = batch_y.to(self.device)
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # タプルの場合は最初の要素を使用
        # print("outputs.shape:",outputs.shape,"batch_y.shape:",batch_y.shape)
        loss = criterion(outputs, batch_y)
        return loss

    def _update(self, batch, criterion, optimizer, scaler=None):
        """
        モデルの更新（逆伝播とパラメータ更新）

        Args:
            batch: 入力バッチデータ
            criterion: 損失関数
            optimizer: 最適化器
            scaler: 自動混合精度用のスケーラー

        Returns:
            tuple: (損失, 出力)
        """
        if not isinstance(optimizer, tuple):
            optimizer = (optimizer, )  # タプルに変換
        for optim in optimizer:
            optim.zero_grad()  # 勾配をクリア
        outputs = self.forward(batch)  # 順伝播
        loss = self.train_loss(criterion, batch, outputs)  # 損失計算
        if self.args.use_amp:
            # 自動混合精度での逆伝播
            scaler.scale(loss).backward()
            for optim in optimizer:
                scaler.step(optim)
            scaler.update()
        else:
            # 通常の逆伝播
            loss.backward()
            for optim in optimizer:
                optim.step()
        return loss, outputs

    def state_dict(self, destination: typing.OrderedDict[str, torch.Tensor]=None,
                   prefix='', keep_vars=False, local_rank=-1) -> typing.OrderedDict[str, torch.Tensor]:
        """
        モデルと最適化器の状態辞書を返す

        Args:
            destination: 出力先の辞書
            prefix: キーのプレフィックス
            keep_vars: 変数を保持するかどうか
            local_rank: 分散学習時のローカルランク

        Returns:
            OrderedDict: モデルと最適化器の状態辞書
        """
        if hasattr(self.args, 'save_opt') and self.args.save_opt:
            # 最適化器の状態も保存する場合
            if destination is None:
                destination = OrderedDict()
                destination._metadata = OrderedDict()
            destination['model'] = self.model.state_dict() if local_rank == -1 else self.model.module.state_dict()
            if hasattr(self.args, 'freeze') and self.args.freeze:
                # 凍結パラメータを除外
                for k, v in self.model.named_parameters() if local_rank == -1 else self.model.module.named_parameters():
                    if not v.requires_grad:
                        destination['model'].pop(k)
            destination['model_optim'] = self.model_optim.state_dict()
            return destination
        else:
            # モデルの状態のみ保存
            return self.model.state_dict() if local_rank == -1 else self.model.module.state_dict()

    def load_state_dict(self, state_dict: typing.OrderedDict[str, torch.Tensor], model=None, strict=False) -> nn.Module:
        """
        状態辞書からモデルと最適化器にパラメータをコピー

        Args:
            state_dict: パラメータとバッファを含む辞書
            model: 対象モデル（Noneの場合はself.model）
            strict: 厳密なロードを行うかどうか

        Returns:
            nn.Module: ロードされたモデル
        """
        if model is None:
            model = self.model
        if 'model_optim' not in state_dict:
            # 最適化器の状態が含まれていない場合
            model.load_state_dict(remove_state_key_prefix(state_dict, model), strict=strict)
        else:
            # 最適化器の状態も含まれている場合
            for k, v in state_dict.items():
                if k == 'model':
                    model.load_state_dict(remove_state_key_prefix(v, model), strict=strict)
                elif hasattr(self, k) and getattr(self, k) is not None:
                    if isinstance(getattr(self, k), optim.Optimizer):
                        # 最適化器の状態をロード
                        assert len(getattr(self, k).param_groups) == len(v['param_groups'])
                        try:
                            getattr(self, k).load_state_dict(v)
                        except ValueError:
                            warnings.warn(f'{k} has different state dict from the checkpoint. '
                                          f'Trying to save all states of frozen parameters...')
                            assert k == 'model_optim'
                            self.model_optim = self._select_optimizer(filter_frozen=False, return_self=False, model=model)
                            self.model_optim.load_state_dict(v)
                            self.remove_frozen_param_from_optim(self.model_optim)
                    else:
                        getattr(self, k).load_state_dict(v, strict=strict)

        return model

    def remove_frozen_param_from_optim(self, model_optim):
        """
        最適化器から凍結パラメータを削除

        Args:
            model_optim: 対象の最適化器
        """
        new_index = []
        for i, p in enumerate(model_optim.param_groups[0]['params']):
            if p.requires_grad:
                new_index.append(i)
        model_optim.param_groups[0]['params'] = [model_optim.param_groups[0]['params'][i] for i in new_index]
        delete_ps = []
        for p in model_optim.state:
            if not p.requires_grad:
                delete_ps.append(p)
        for p in delete_ps:
            model_optim.state.pop(p)

    def load_checkpoint(self, load_path=None, model=None, strict=False):
        """
        チェックポイントファイルからモデルをロード

        Args:
            load_path: チェックポイントファイルのパス
            model: 対象モデル
            strict: 厳密なロードを行うかどうか

        Returns:
            nn.Module: ロードされたモデル
        """
        return self.load_state_dict(torch.load(load_path, map_location=self.device), model, strict=strict)

    def vali(self):
        """検証処理（サブクラスで実装）"""
        pass

    def train(self):
        """訓練処理（サブクラスで実装）"""
        pass

    def test(self):
        """テスト処理（サブクラスで実装）"""
        pass

"""
Proceed実験クラス
Proceed手法の実験を実行するクラス
Exp_Onlineを継承してProceed特有の処理を実装

Proceed手法の特徴:
- アダプター層を使用した効率的なオンライン学習
- 最近のデータバッチを保持して適応的な更新
- バックボーンネットワークの凍結/解凍による段階的学習
"""

import copy
import numpy as np
import torch
from tqdm import tqdm

from tool_proceed import proceed
from data_provider.data_factory import get_dataset, get_dataloader
from data_provider.data_loader import Dataset_Recent
from exp import Exp_Online


class Exp_Proceed(Exp_Online):
    """
    Proceed手法の実験クラス

    Proceed手法は、アダプター層を使用して効率的なオンライン学習を実現する手法です。
    バックボーンネットワークを凍結し、アダプター層のみを更新することで、
    計算効率を保ちながら新しいデータに適応します。
    """

    def __init__(self, args):
        """
        Proceed実験クラスの初期化

        Args:
            args: 実験設定パラメータ
        """
        # 引数をディープコピーして変更可能にする
        args = copy.deepcopy(args)
        # 重みマージを有効化（Proceed特有の設定）
        args.merge_weights = 1
        # 親クラスの初期化
        super(Exp_Proceed, self).__init__(args)
        # オンライン学習フェーズの定義
        self.online_phases = ['val', 'test', 'online']
        # 平均化の次元（シーケンス長を使用）
        self.mean_dim = args.seq_len

    def online(self, online_data=None, target_variate=None, phase='test', show_progress=False):
        """
        オンライン学習の実行

        Args:
            online_data: オンライン学習用データ
            target_variate: 予測対象変数
            phase: 学習フェーズ（'val', 'test', 'online'）
            show_progress: プログレスバーの表示フラグ

        Returns:
            オンライン学習の結果
        """
        # 検証フェーズでオンライン学習率を使用する場合の処理
        if phase == 'val' and self.args.val_online_lr:
            # 現在の学習率を保存
            lr = self.model_optim.param_groups[0]['lr']
            # 全パラメータグループの学習率をオンライン学習率に変更
            for j in range(len(self.model_optim.param_groups)):
                self.model_optim.param_groups[j]['lr'] = self.args.online_learning_rate

        # 勾配をゼロクリア
        self.model_optim.zero_grad()
        # アダプター層を凍結（バックボーンのみ更新）
        self._model.freeze_adapter(True)
        # 親クラスのオンライン学習を実行
        ret = super().online(online_data, target_variate, phase, show_progress)
        # アダプター層の凍結を解除
        self._model.freeze_adapter(False)

        # 検証フェーズで学習率を元に戻す
        if phase == 'val' and self.args.val_online_lr:
            for j in range(len(self.model_optim.param_groups)):
                self.model_optim.param_groups[j]['lr'] = lr
        return ret

    def update_valid(self, valid_data=None, valid_dataloader=None):
        """
        検証データを使用したモデル更新

        このメソッドは、検証データを使用してモデルを段階的に更新します。
        最近のデータバッチと現在のデータバッチを分けて処理し、
        アダプター層とバックボーンの更新を制御します。

        Args:
            valid_data: 検証データ
            valid_dataloader: 検証データローダー

        Returns:
            予測結果のリスト
        """
        # フェーズを'online'に設定
        self.phase = 'online'

        # 検証データが指定されていない場合、データセットを取得
        if valid_data is None:
            valid_data = get_dataset(self.args, 'val', self.device, wrap_class=self.args.wrap_data_class + [Dataset_Recent], **self.wrap_data_kwargs, take_post=self.args.pred_len - 1)

        # オンライン学習用のデータローダーを作成
        valid_loader = get_dataloader(valid_data, self.args, 'online')
        # オプティマイザーと損失関数を選択
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        # 混合精度学習用のスケーラー
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        # モデルを学習モードに設定
        self.model.train()
        predictions = []

        # 結合更新が無効な場合の処理
        if not self.args.joint_update_valid:
            for i, (recent_batch, current_batch) in enumerate(tqdm(valid_loader, mininterval=10)):
                # バイアスパラメータの凍結を解除
                self._model.freeze_bias(False)
                # アダプター層を凍結（バックボーンのみ更新）
                self._model.freeze_adapter(True)
                # 最近のデータバッチでオンライン更新（バックボーン更新）
                self._update_online(recent_batch, criterion, model_optim, scaler, flag_current=False)

                # バイアスパラメータを凍結
                self._model.freeze_bias(True)
                # バックボーンを凍結（アダプター層のみ更新）
                if not self.args.freeze:
                    self._model.backbone.requires_grad_(False)
                # アダプター層の凍結を解除
                self._model.freeze_adapter(False)
                # 現在のデータバッチでオンライン更新（アダプター層更新）
                _, outputs = self._update_online(current_batch, criterion, model_optim, scaler, flag_current=True)
                # バックボーンの凍結を解除
                if not self.args.freeze:
                    self._model.backbone.requires_grad_(True)

                # 予測結果を保存
                if self.args.do_predict:
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    predictions.append(outputs.detach().cpu().numpy())

            # バイアスパラメータの凍結を解除
            self._model.freeze_bias(False)

        # 結合更新が有効な場合の処理
        else:
            for i, (recent_batch, current_batch) in enumerate(tqdm(valid_loader, mininterval=10)):
                # 最近のデータバッチでオンライン更新
                self._update_online(recent_batch, criterion, model_optim, scaler, flag_current=True)

                # 予測結果を保存
                if self.args.do_predict:
                    self.model.eval()
                    with torch.no_grad():
                        outputs = self.forward(current_batch)
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    predictions.append(outputs.detach().cpu().numpy())
                    self.model.train()

        # 勾配をゼロクリア
        self.model_optim.zero_grad()
        # アダプター層を凍結
        self._model.freeze_adapter(True)

        # 学習可能パラメータ数を計算・表示
        trainable_params = sum([param.nelement() if param.requires_grad else 0 for param in self._model.parameters()])
        print(f'Trainable Params: {trainable_params}', '({:.1f}%)'.format(trainable_params / self.model_params * 100))

        return predictions

    def _build_model(self, model=None, framework_class=None):
        """
        モデルの構築

        Args:
            model: 既存のモデル
            framework_class: フレームワーククラス（Proceedを使用）

        Returns:
            構築されたモデル
        """
        # Proceedフレームワーククラスを使用してモデルを構築
        model = super()._build_model(model, framework_class= proceed.Proceed)
        print(model)
        return model

    def _update(self, batch, criterion, optimizer, scaler=None):
        """
        モデル更新処理

        Args:
            batch: 入力バッチ
            criterion: 損失関数
            optimizer: オプティマイザー
            scaler: 混合精度学習用スケーラー

        Returns:
            損失値と出力
        """
        # 更新フラグを設定
        self._model.flag_update = True
        # 親クラスの更新処理を実行
        loss, outputs = super()._update(batch, criterion, optimizer, scaler)
        # 最近のバッチを保存（入力と出力を結合）
        self._model.recent_batch = torch.cat([batch[0], batch[1]], -2)
        # 更新フラグをリセット
        self._model.flag_update = False
        return loss, outputs

    def _update_online(self, batch, criterion, optimizer, scaler=None, flag_current=False):
        """
        オンライン更新処理

        Args:
            batch: 入力バッチ
            criterion: 損失関数
            optimizer: オプティマイザー
            scaler: 混合精度学習用スケーラー
            flag_current: 現在のデータフラグ

        Returns:
            損失値と出力
        """
        # オンライン学習フラグを設定
        self._model.flag_online_learning = True
        self._model.flag_current = flag_current
        # 親クラスのオンライン更新処理を実行
        loss, outputs = super()._update_online(batch, criterion, optimizer, scaler)
        # 最近のバッチを保存
        self._model.recent_batch = torch.cat([batch[0], batch[1]], -2)
        # フラグをリセット
        self._model.flag_online_learning = False
        self._model.flag_current = not flag_current
        return loss, outputs

    def analysis_online(self):
        """
        オンライン学習の分析

        Returns:
            オンライン学習の分析結果
        """
        # アダプター層を凍結して分析を実行
        self._model.freeze_adapter(True)
        return super().analysis_online()

    def predict(self, path, setting, load=False):
        """
        予測の実行

        Args:
            path: モデルパス
            setting: 設定名
            load: モデル読み込みフラグ

        Returns:
            予測結果
        """
        # 検証データでモデルを更新
        self.update_valid()
        # オンライン学習を実行
        res = self.online()
        # 予測結果を保存
        np.save('./results/' + setting + '_pred.npy', np.vstack(res[-1]))
        return res
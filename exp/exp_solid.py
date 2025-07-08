"""
SOLID実験クラス
SOLID（Selective Online Learning with Incremental Data）手法の実験を実行するクラス
Exp_Onlineを継承してSOLID特有の処理を実装

SOLID手法の特徴:
- 類似度に基づく選択的オンライン学習
- 周期的なデータパターンを考慮したサンプル選択
- 効率的なメモリ使用と計算コストの削減
- 事前学習済みモデルの知識を保持しながら新しいデータに適応
"""

import time
import numpy as np
from tqdm import tqdm
import warnings
import copy

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import settings
from exp import Exp_Online
from data_provider.data_factory import data_provider, get_dataset
from util.metrics import update_metrics, calculate_metrics
from util.tools import test_params_flop

warnings.filterwarnings('ignore')

def get_period(dataset_name):
    """
    データセット名から周期性を取得する関数

    各データセットの周期性（時間間隔）を定義します。
    これはSOLID手法で類似サンプルを選択する際の重要なパラメータです。

    Args:
        dataset_name: データセット名

    Returns:
        period: データセットの周期性（時間間隔）
    """
    if "ETTh1" in dataset_name:
        period = 24  # 1時間間隔、24時間周期
    elif "ETTh2" in dataset_name:
        period = 24  # 1時間間隔、24時間周期
        # period = 1  # 1時間間隔、24時間周期
    elif "ETTm1" in dataset_name:
        period = 96  # 15分間隔、24時間周期
    elif "ETTm2" in dataset_name:
        period = 96  # 15分間隔、24時間周期
    elif "electricity" in dataset_name:
        period = 24  # 1時間間隔、24時間周期
    elif "ECL" in dataset_name:
        period = 24  # 1時間間隔、24時間周期
    elif "traffic" in dataset_name.lower():
        period = 24  # 1時間間隔、24時間周期
    elif "illness" in dataset_name.lower():
        period = 52.142857  # 週次データ
    elif "weather" in dataset_name.lower():
        period = 144  # 10分間隔、24時間周期
    elif "Exchange" in dataset_name:
        period = 1  # 日次データ
    elif "WTH_informer" in dataset_name:
        period = 24  # 1時間間隔、24時間周期
    else:
        period = 1  # デフォルト値
    return period


class Exp_SOLID(Exp_Online):
    """
    SOLID手法の実験クラス

    SOLID手法は、類似度に基づいて過去のサンプルを選択的に使用し、
    効率的なオンライン学習を実現する手法です。周期的なデータパターンを
    考慮して、最も関連性の高いサンプルを選択します。
    """

    def __init__(self, args):
        """
        SOLID実験クラスの初期化

        Args:
            args: 実験設定パラメータ
        """
        # 親クラスの初期化
        super(Exp_SOLID, self).__init__(args)

        # 表現ベクトルのパス設定（事前計算された表現を使用する場合）
        self.rep_path = args.rep_path if hasattr(args, 'use_rep') and args.use_pred and hasattr(args, 'rep_path') else None

        # 基本的なパラメータの設定
        self.seq_len = args.seq_len  # 入力シーケンス長
        self.pred_len = args.pred_len  # 予測長
        self.label_len = args.label_len  # Transformer用のラベル長
        self.buffer_size = args.buffer_size  # テスト時の学習サンプル数

        # 全モデル更新か部分更新かの設定
        if not args.whole_model:
            # 部分更新の場合、メインモデルを凍結
            self.model.requires_grad_(False)
        else:
            # 全モデル更新の場合、学習率をオンライン学習率に設定
            for j in range(len(self.model_optim.param_groups)):
                self.model_optim.param_groups[j]['lr'] = args.online_learning_rate

        # データセットの周期性を取得
        self.period = get_period(self.args.dataset)

        # モデル別の線形層名マッピング
        linear_map = {
                "Linear": "Linear",
                "NLinear": "Linear",
                "PatchTST": "model.head.linear",
                "TCN": "regressor",
                "iTransformer": "projector",
                "default": "decoder.projection",
        }
        self.linear_name = linear_map[self.args.model] if self.args.model in linear_map else linear_map["default"]

        # 正規化が有効な場合、バックボーン名を付加
        if self.args.normalization:
            self.linear_name = 'backbone.' + self.linear_name

        # 最終的な更新対象のヘッドを設定
        if self.args.whole_model:
            self.final_head = self.model  # 全モデル更新
        else:
            self.final_head = self.model.get_submodule(self.linear_name)  # 部分更新
            self.final_head.requires_grad_(True)

        # 周期的なインデックスの計算
        # 予測時刻から過去のサンプルを周期的に選択するためのインデックス
        indices = torch.arange(self.pred_len + self.buffer_size - 1, self.pred_len - 1, step=-1) % self.period
        threshold = self.period * self.args.lambda_period  # 周期性の閾値
        self.indices = torch.arange(self.buffer_size)[(indices <= threshold) & (indices >= -threshold)]

        # 入力データのインデックス
        self.indices_x = (self.indices.unsqueeze(-1) + torch.arange(self.seq_len)).unsqueeze(-1).expand(-1, -1, self.args.enc_in)
        # 出力データのインデックス
        self.indices_y = (self.indices.unsqueeze(-1) + torch.arange(self.seq_len, self.seq_len + self.pred_len)).unsqueeze(-1).expand(-1, -1, self.args.enc_in)
        # 時間特徴量のインデックス（x）
        self.indices_x_mark = (self.indices.unsqueeze(-1) + torch.arange(self.seq_len)).unsqueeze(-1)
        # 時間特徴量のインデックス（y）
        self.indices_y_mark = (self.indices.unsqueeze(-1) + torch.arange(self.seq_len - self.label_len, self.seq_len + self.pred_len)).unsqueeze(-1)

        # インデックスをデバイスに移動
        self.indices = self.indices.to(self.device)
        self.indices_x = self.indices_x.to(self.device)
        self.indices_x_mark = self.indices_x_mark.to(self.device)
        self.indices_y_mark = self.indices_y_mark.to(self.device)
        self.indices_y = self.indices_y.to(self.device)

        # バッチサイズの設定（メモリ効率のため）
        self.batch_size = -1
        if self.args.dataset == 'Traffic' and self.args.mini_batch > 20:
            self.batch_size = 20
        elif self.args.dataset == 'ECL' and self.args.mini_batch > 40:
            self.batch_size = 20

    def _select_optimizer(self, *args, **kwargs):
        """
        オプティマイザーの選択

        全モデル更新の場合のみオプティマイザーを返し、
        部分更新の場合はNoneを返します。

        Returns:
            オプティマイザーまたはNone
        """
        if self.args.whole_model:
            return super()._select_optimizer(*args, **kwargs)
        return None

    def _forward(self, x, rep):
        """
        順伝播処理

        表現ベクトルから予測を生成します。モデルによって
        異なる後処理を適用します。

        Args:
            x: 入力データ
            rep: 表現ベクトル

        Returns:
            予測結果
        """
        # 表現ベクトルから予測を生成
        pred = self.final_head(rep.detach())

        # モデル別の後処理
        if self.model == 'PatchTST' and self.args.revin:
            # PatchTST + RevINの場合
            pred = pred.permute(0,2,1)
            pred = self.model.backbone.revin_layer(pred, 'denorm')
        elif self.model == 'iTransformer':
            # iTransformerの場合、正規化の逆変換
            means = x.mean(1, keepdim=True).detach()
            stdev = torch.sqrt(torch.var(x - means, dim=1, keepdim=True, unbiased=False) + 1e-5)
            pred = pred * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            pred = pred + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        elif self.model == 'TCN':
            # TCNの場合、形状を調整
            pred = pred.reshape(len(pred), self.pred_len, -1)

        # RevIN正規化の逆変換
        if self.args.normalization.lower() == 'revin':
            pred = self.model.processor(pred, 'denorm')

        return pred

    def online(self, online_data=None, target_variate=None, phase='test', show_progress=False):
        """
        オンライン学習の実行

        SOLID手法の核心部分。類似度に基づいて過去のサンプルを選択し、
        選択されたサンプルでモデルを更新してから予測を行います。

        Args:
            online_data: オンライン学習用データ
            target_variate: 予測対象変数
            phase: 学習フェーズ
            show_progress: プログレスバーの表示フラグ

        Returns:
            予測結果とメトリクス
        """
        predictions = []
        self.model.eval()

        # オンラインデータの準備
        if online_data is None:
            online_data = get_dataset(self.args, 'test', self.device, take_pre=self.buffer_size + self.pred_len - 1, wrap_class=self.args.wrap_data_class, **self.wrap_data_kwargs)

        # データ境界の確認
        assert self.args.borders[1][0] - self.buffer_size - self.pred_len + 1 >= 0

        # 事前計算された表現ベクトルの読み込み
        if self.rep_path:
            all_reps = np.load(self.rep_path)
            assert len(all_reps) == self.args.borders[1][1] - self.seq_len - self.pred_len + 1
            all_reps = all_reps[self.args.borders[1][0] - self.buffer_size - self.pred_len + 1:]

        # データの取得
        data_x = online_data.data_x
        data_y = online_data.data_y
        data_ts = online_data.data_stamp.to(data_x.device)

        # 時間特徴量の次元調整
        if self.indices_x_mark.shape[-1] != data_ts.shape[-1]:
            self.indices_x_mark = self.indices_x_mark.expand(-1, -1, data_ts.shape[-1])
            self.indices_y_mark = self.indices_y_mark.expand(-1, -1, data_ts.shape[-1])

        # テスト用データローダーの作成
        test_loader = DataLoader(online_data, batch_size=1, shuffle=False, num_workers=self.args.num_workers, drop_last=False, pin_memory=False)

        # 損失関数と統計情報の初期化
        criterion = nn.MSELoss()
        statistics = {k: 0 for k in ['total', 'y_sum', 'MSE', 'MAE']}

        # 継続学習でない場合の事前学習済み状態の保存
        if not self.args.continual:
            if not self.args.whole_model:
                pretrained_state_dict = copy.deepcopy(self.final_head.state_dict())
            else:
                pretrained_state_dict = copy.deepcopy(self.state_dict())

        # オンライン学習ループ
        for i, batch in enumerate(tqdm(test_loader, mininterval=10)):
            # 初期サンプルはスキップ（十分な履歴データがないため）
            if i < self.buffer_size + self.pred_len - 1:
                continue

            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            batch_x = batch_x.to(self.device)

            # 時間インデックスの計算
            t = i + self.seq_len
            start = i - self.pred_len - self.buffer_size + 1

            # 過去のデータから周期的なサンプルを取得
            lookback_x = data_x[start: t].expand(len(self.indices), -1, -1).gather(1, self.indices_x)

            # サンプル類似度の計算（ユークリッド距離）
            distance_pairs = F.pairwise_distance(batch_x.view(-1), lookback_x.view(len(lookback_x), -1), p=2)
            # 最も類似度の高いサンプルを選択
            selected_indices = distance_pairs.topk(self.args.mini_batch, largest=False)[1]
            idx = self.indices[selected_indices]
            selected_x = lookback_x[selected_indices]
            selected_y = data_y[start: t].expand(len(idx), -1, -1).gather(1, self.indices_y[selected_indices])

            # 表現ベクトルまたは時間特徴量の取得
            if self.rep_path:
                selected_reps = all_reps[i - (self.buffer_size + self.pred_len - 1) + idx]
                # selected_pred = self._forward(selected_x, selected_reps)
            else:
                selected_x_mark = data_ts[start: t].expand(len(idx), -1, -1).gather(1, self.indices_x_mark[selected_indices])
                selected_y_mark = data_ts[start: t].expand(len(idx), -1, -1).gather(1, self.indices_y_mark[selected_indices])

            # モデルの更新
            self.model.train()
            if not self.args.whole_model:
                # 部分更新の場合（最終層のみ更新）
                selected_pred = self.forward([selected_x, selected_y, selected_x_mark, selected_y_mark])
                if isinstance(selected_pred, tuple):
                    selected_pred = selected_pred[0]
                pretrained_state_dict = copy.deepcopy(self.final_head.state_dict())
                model_optim = optim.SGD(self.final_head.parameters(), lr=self.args.online_learning_rate)

                loss = criterion(selected_pred, selected_y)
                loss.backward()
                model_optim.step()
                model_optim.zero_grad()
            else:
                # 全モデル更新の場合
                self._update([selected_x, selected_y, selected_x_mark, selected_y_mark], criterion, self.model_optim)
                self.model_optim.zero_grad()

            # 予測の実行
            with torch.no_grad():
                self.model.eval()
                outputs = self.forward(batch)
                if self.args.do_predict:
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    predictions.append(outputs.detach().cpu().numpy())

                # メトリクスの更新
                true = batch[self.label_position]
                if not self.args.pin_gpu:
                    true = true.to(self.device)
                update_metrics(outputs, true, statistics, target_variate)

            # 継続学習でない場合、事前学習済み状態に戻す
            if not self.args.continual:
                if not self.args.whole_model:
                    self.final_head.load_state_dict(pretrained_state_dict)
                else:
                    self.load_state_dict(pretrained_state_dict)

        # 最終メトリクスの計算
        metrics = calculate_metrics(statistics)
        mse, mae = metrics['MSE'], metrics['MAE']
        print('mse:{}, mae:{}'.format(mse, mae))

        if self.args.do_predict:
            return mse, mae, online_data, predictions
        else:
            return mse, mae, online_data

    def analysis_online(self):
        """
        オンライン学習の分析

        オンライン学習の性能分析を行います。更新時間と推論時間を計測し、
        メモリ使用量も確認します。

        Returns:
            分析結果
        """
        self.model.eval()

        # オンラインデータの準備
        online_data = get_dataset(self.args, 'test', self.device,
                                  take_pre=self.buffer_size + self.pred_len - 1,
                                  wrap_class=self.args.wrap_data_class, **self.wrap_data_kwargs)

        # データ境界の確認
        assert self.args.borders[1][0] - self.buffer_size - self.pred_len + 1 >= 0

        # データの取得
        data_x = online_data.data_x.to(self.device)
        data_y = online_data.data_y.to(self.device)
        data_ts = online_data.data_stamp.to(data_x.device)

        # 時間特徴量の次元調整
        if self.indices_x_mark.shape[-1] != data_ts.shape[-1]:
            self.indices_x_mark = self.indices_x_mark.expand(-1, -1, data_ts.shape[-1])
            self.indices_y_mark = self.indices_y_mark.expand(-1, -1, data_ts.shape[-1])

        # テスト用データローダーの作成
        test_loader = DataLoader(online_data, batch_size=1, shuffle=False, num_workers=self.args.num_workers,
                                 drop_last=False, pin_memory=False)

        # 損失関数と統計情報の初期化
        criterion = nn.MSELoss()
        statistics = {k: 0 for k in ['total', 'y_sum', 'MSE', 'MAE']}

        # 継続学習でない場合の事前学習済み状態の保存
        if not self.args.continual:
            if not self.args.whole_model:
                pretrained_state_dict = copy.deepcopy(self.final_head.state_dict())
            else:
                pretrained_state_dict = copy.deepcopy(self.state_dict())

        # 時間計測用のリスト
        times_update = []
        times_infer = []
        print('GPU Mem:', torch.cuda.max_memory_allocated())

        j = 0
        for i, batch in enumerate(tqdm(test_loader, mininterval=10)):

            # 初期サンプルはスキップ
            if i < self.buffer_size + self.pred_len - 1:
                continue

            # 分析用の制限（50サンプルまで）
            if j == 50:
                break

            start_time = time.time()
            batch = [d.to(self.device) for d in batch]
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch

            # 時間インデックスの計算
            t = i + self.seq_len
            start = i - self.pred_len - self.buffer_size + 1

            # 過去のデータから周期的なサンプルを取得
            lookback_x = data_x[start: t].expand(len(self.indices), -1, -1).gather(1, self.indices_x)

            # サンプル類似度の計算と選択
            distance_pairs = F.pairwise_distance(batch_x.view(-1), lookback_x.view(len(lookback_x), -1), p=2)
            selected_indices = distance_pairs.topk(self.args.mini_batch, largest=False)[1]
            idx = self.indices[selected_indices]
            selected_x = lookback_x[selected_indices]
            selected_y = data_y[start: t].expand(len(idx), -1, -1).gather(1, self.indices_y[selected_indices])
            selected_x_mark = data_ts[start: t].expand(len(idx), -1, -1).gather(1, self.indices_x_mark[selected_indices])
            selected_y_mark = data_ts[start: t].expand(len(idx), -1, -1).gather(1, self.indices_y_mark[selected_indices])

            # デバッグ用の出力
            if j == 5:
                print(selected_x.shape)

            # モデルの更新
            self.model.train()
            if not self.args.whole_model:
                # 部分更新の場合
                pretrained_state_dict = copy.deepcopy(self.final_head.state_dict())
                model_optim = optim.SGD(self.final_head.parameters(), lr=self.args.online_learning_rate)
                for ii in range(len(selected_x)):
                    selected_pred = self.forward([selected_x[[ii]], selected_y[[ii]], selected_x_mark[[ii]], selected_y_mark[[ii]]])
                    if isinstance(selected_pred, tuple):
                        selected_pred = selected_pred[0]
                    loss = criterion(selected_pred, selected_y[[ii]])
                    loss.backward()
                model_optim.step()
                model_optim.zero_grad()
            else:
                # 全モデル更新の場合
                for ii in range(len(selected_x)):
                    selected_pred = self.forward([selected_x[[ii]], selected_y[[ii]], selected_x_mark[[ii]], selected_y_mark[[ii]]])
                    if isinstance(selected_pred, tuple):
                        selected_pred = selected_pred[0]
                    loss = criterion(selected_pred, selected_y[[ii]])
                    loss.backward()
                self.model_optim.step()
                self.model_optim.zero_grad()

            # 更新時間の記録（10サンプル以降）
            if j > 10:
                times_update.append(time.time() - start_time)

            # 推論時間の計測
            with torch.no_grad():
                start_time = time.time()
                self.model.eval()
                outputs = self.forward(batch)
                if j > 10:
                    times_infer.append(time.time() - start_time)
                if j == 50:
                    break
            j += 1

            # 継続学習でない場合、事前学習済み状態に戻す
            if not self.args.continual:
                if not self.args.whole_model:
                    self.final_head.load_state_dict(pretrained_state_dict)
                else:
                    self.load_state_dict(pretrained_state_dict)

        # 結果の出力
        print('Final GPU Mem:', torch.cuda.max_memory_allocated())
        # 平均時間の計算（最小値と最大値を除外）
        times_update = (sum(times_update) - min(times_update) - max(times_update)) / (len(times_update) - 2)
        times_infer = (sum(times_infer) - min(times_infer) - max(times_infer)) / (len(times_infer) - 2)
        print('Update Time:', times_update)
        print('Infer Time:', times_infer)
        print('Latency:', times_update + times_infer)

        # モデルのパラメータ数とFLOPsの計測
        test_params_flop(self.model, (1, self.args.seq_len, self.args.enc_in))

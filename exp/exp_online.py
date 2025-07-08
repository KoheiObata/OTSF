"""
オンライン学習実験クラス
オンライン時系列予測実験を実行するクラス
Exp_Mainを継承してオンライン学習特有の処理を実装
"""

import copy
from tqdm import tqdm
import time
import warnings

import torch
import torch.distributed as dist

from data_provider.data_factory import data_provider, get_dataset, get_dataloader
from data_provider.data_loader import Dataset_Recent
from exp.exp_main import Exp_Main
from util.metrics import metric, update_metrics, calculate_metrics
from util.tools import test_params_flop



warnings.filterwarnings('ignore')

transformers = ['Autoformer', 'Transformer', 'Informer']

# =============================
# オンライン学習の基底クラス
# =============================
class Exp_Online(Exp_Main):
    """
    オンライン時系列予測実験の基底クラス
    Exp_Mainを継承し、オンライン学習特有の処理を実装
    train()はExp_Mainから継承する，主にvaliやtestにおけるオンライン学習の実装

    主な機能：
    - オンライン学習フェーズ（test, online）でのデータ取得
    - 情報リークあり/なしのオンライン学習
    - 逐次的なモデル更新
    - オンライン学習の性能評価
    """
    def __init__(self, args):
        """
        初期化処理
        - オンライン学習フェーズの設定
        - 逐次学習用のデータ取得設定
        """
        super().__init__(args)
        # オンライン学習で使うフェーズ名
        self.online_phases = ['test', 'online']
        # 逐次学習時のデータ取得設定
        self.wrap_data_kwargs.update(recent_num=1, gap=self.args.pred_len)

    def _get_data(self, flag, **kwargs):
        """
        オンライン学習フェーズの場合のデータ取得
        - flag: データセットの種類（'train', 'val', 'test', 'online'）
        - leakage=True: 情報リークあり（未来の正解を使って即時更新）
        - leakage=False: 情報リークなし（正しいタイミングでのみ更新）
        """
        # オンライン学習フェーズの場合のデータ取得
        if flag in self.online_phases:
            # 情報リークあり（leakage=True）の場合
            if self.args.leakage:
                # 未来の正解を使って即時更新するデータセット
                data_set = get_dataset(self.args, flag, self.device, wrap_class=self.args.wrap_data_class, **kwargs)
                data_loader = get_dataloader(data_set, self.args, 'online' if flag == 'test' else 'test')
            else:
                # 情報リークなし（正しいタイミングでのみ更新）
                data_set = get_dataset(self.args, flag, self.device, wrap_class=self.args.wrap_data_class + [Dataset_Recent], **self.wrap_data_kwargs, **kwargs)
                data_loader = get_dataloader(data_set, self.args, 'online')
            return data_set, data_loader
        else:
            # 通常のデータ取得
            return super()._get_data(flag, **kwargs)

    def vali(self, vali_data, vali_loader, criterion):
        """
        バリデーション処理
        - 情報リークあり or valフェーズがオンラインでない場合は通常のバリデーション
        - 情報リークなしの場合はオンラインバリデーション
        """
        self.phase = 'val'
        # 情報リークあり or valフェーズがオンラインでない場合は通常のバリデーション
        if self.args.leakage or 'val' not in self.online_phases:
            mse = super().vali(vali_data, vali_loader, criterion)
        else:
            # オンラインバリデーション（情報リークなし）
            if self.args.local_rank <= 0:
                state_dict = copy.deepcopy(self.state_dict())
                mse = self.online(online_data=vali_data, target_variate=None, phase='val')[0]
                if self.args.local_rank == 0:
                    mse = torch.tensor(mse, device=self.device)
                self.load_state_dict(state_dict, strict=not (hasattr(self.args, 'freeze') and self.args.freeze))
            else:
                mse = torch.tensor(0, device=self.device)
            if self.args.local_rank >= 0:
                dist.all_reduce(mse, op=dist.ReduceOp.SUM)
                mse = mse.item()
        return mse

    def update_valid(self, valid_data=None):
        """
        バリデーションデータでのオンライン更新
        - 情報リークあり/なしで処理を分岐
        - 逐次的にモデルを更新
        """
        self.phase = 'online'
        # =============================
        # 情報リークあり（leakage=True）の場合
        # =============================
        if hasattr(self.args, 'leakage') and self.args.leakage:
            valid_data = get_dataset(self.args, 'val', self.device, wrap_class=self.args.wrap_data_class, take_pre=True, take_post=self.args.pred_len - 1, **self.wrap_data_kwargs)
            self.online_information_leakage(valid_data, None, 'online', True)
            return []

        # =============================
        # 情報リークなし（leakage=False）の場合
        # =============================
        if valid_data is None or not isinstance(valid_data, Dataset_Recent):
            valid_data = get_dataset(self.args, 'val', self.device, wrap_class=self.args.wrap_data_class + [Dataset_Recent], take_post=self.args.pred_len - 1, **self.wrap_data_kwargs)
        valid_loader = get_dataloader(valid_data, self.args, flag='online')

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None
        predictions = []
        for i, (recent_batch, current_batch) in enumerate(tqdm(valid_loader, mininterval=10)):
            self.model.train()
            # 逐次的にモデルをオンライン更新
            self._update_online(recent_batch, criterion, model_optim, scaler)
            if self.args.do_predict:
                self.model.eval()
                with torch.no_grad():
                    outputs = self.forward(current_batch)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                predictions.append(outputs.detach().cpu().numpy())
        return predictions

    def _update_online(self, batch, criterion, optimizer, scaler=None):
        """
        オンライン学習1ステップ分のパラメータ更新処理
        - 通常のバッチ処理または逐次処理に対応
        - 複数オプティマイザーにも対応
        - batch: 入力データ（通常は[seq_x, seq_y, ...]のリストやタプル）
        - criterion: 損失関数
        - optimizer: オプティマイザー（またはそのタプル）
        - scaler: AMP用スケーラー（省略可）
        戻り値: (loss, outputs)
        """
        # バッチの最初の要素が3次元（通常のバッチ）なら通常の_updateを呼ぶ
        if batch[0].dim() == 3:
            # 通常のバッチ学習（複数系列をまとめて一度に更新）
            return self._update(batch, criterion, optimizer, scaler)
        else:
            #　使わないと思ったから消した．必要なら元のコードを復活させる
            warnings.warn("逐次処理は未実装")
            exit()

    def online(self, online_data=None, target_variate=None, phase='test', show_progress=False):
        """
        オンライン学習のメインループ
        - 情報リークあり/なしで処理を分岐
        - 逐次的にモデルを更新しながら予測
        - 性能指標（MSE, MAE）を計算
        - online_data: オンライン学習用データセット（省略時は自動生成）
        - target_variate: 評価対象変数（省略可）
        - phase: 'test' or 'val' or 'online' など
        - show_progress: tqdmによる進捗表示
        戻り値: (mse, mae, online_data, [predictions])
        """
        self.phase = phase
        # =============================
        # 情報リークあり（leakage=True）の場合
        # =============================
        if hasattr(self.args, 'leakage') and self.args.leakage:
            online_data = get_dataset(self.args, phase, self.device, wrap_class=self.args.wrap_data_class, **self.wrap_data_kwargs)
            return self.online_information_leakage(online_data, target_variate, phase, show_progress)

        # =============================
        # 情報リークなし（leakage=False）の場合
        # =============================
        # データセットが未指定なら自動生成
        if online_data is None:
            online_data = get_dataset(self.args, phase, self.device, wrap_class=self.args.wrap_data_class + [Dataset_Recent], **self.wrap_data_kwargs)
        # DataLoaderを取得
        online_loader = get_dataloader(online_data, self.args, flag='online')

        if self.args.do_predict:
            predictions = []  # 予測結果を格納
        # 性能指標の累積用辞書
        statistics = {k: 0 for k in ['total', 'y_sum', 'MSE', 'MAE']}

        # オプティマイザー・損失関数・AMPスケーラーを準備
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        # 進捗表示（tqdmで進捗バー表示）
        if show_progress:
            online_loader = tqdm(online_loader, mininterval=10)
        # オンライン学習のメインループ
        for i, (recent_data, current_data) in enumerate(online_loader):
            self.model.train()  # モデルを訓練モードに
            # 逐次的にモデルをオンライン更新（recent_dataでパラメータ更新）
            self._update_online(recent_data, criterion, model_optim, scaler)
            self.model.eval()  # モデルを推論モードに
            with torch.no_grad():
                # current_dataで予測
                outputs = self.forward(current_data)
                if self.args.do_predict:
                    # 予測結果を保存
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    predictions.append(outputs.detach().cpu().numpy())
                # 性能指標（MSE, MAEなど）を更新 #label_position=1（これはどういう意味なのか？）
                update_metrics(outputs, current_data[self.label_position].to(self.device), statistics, target_variate)


        # 全サンプルの性能指標を集計
        metrics = calculate_metrics(statistics)
        mse, mae = metrics['MSE'], metrics['MAE']
        if phase == 'test':
            print('mse:{}, mae:{}'.format(mse, mae))
        if self.args.do_predict:
            return mse, mae, online_data, predictions
        else:
            return mse, mae, online_data

    # --- 以下、情報リークありのオンライン学習用メソッド ---
    def online_information_leakage(self, online_data=None, target_variate=None, phase='test', show_progress=False):
        """
        情報リークあり：予測直後に正解値で即時更新
        - 現実的には不可能だが、理論的な性能上限を測るための実験
        - 予測直後に正解値を使って即座にモデルを更新する
        - online_data: オンライン学習用データセット（省略時は自動生成）
        - target_variate: 評価対象変数（省略可）
        - phase: 'test' or 'val' or 'online' など
        - show_progress: tqdmによる進捗表示
        戻り値: (mse, mae, online_data, [predictions])
        """
        # データセットが未指定なら自動生成
        if online_data is None:
            online_data = get_dataset(self.args, phase, self.device, wrap_class=self.args.wrap_data_class, **self.wrap_data_kwargs)
        # DataLoaderを取得
        online_loader = get_dataloader(online_data, self.args, flag='online')

        if self.args.do_predict:
            predictions = []  # 予測結果を格納
        # 性能指標の累積用辞書
        statistics = {k: 0 for k in ['total', 'y_sum', 'MSE', 'MAE']}

        # オプティマイザー・損失関数・AMPスケーラーを準備
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        # 進捗表示（tqdmで進捗バー表示）
        if show_progress:
            online_loader = tqdm(online_loader, mininterval=10)

        # オンライン学習のメインループ
        for i, current_data in enumerate(online_loader):
            # PatchTSTなどdropoutが重要な場合は無効化して予測する
            if self.args.model == 'PatchTST':
                # online学習でbackwardするときはevalにしてdropoutを無効化した方がいい．
                self.model.eval()  # モデルを推論モードに
                # 予測直後に正解値で即時更新（情報リーク）
                loss, outputs = self._update_online(current_data, criterion, model_optim, scaler)
            else:
                self.model.train()  # モデルを訓練モードに
                # 予測直後に正解値で即時更新（情報リーク）
                loss, outputs = self._update_online(current_data, criterion, model_optim, scaler)
            # 性能指標（MSE, MAEなど）を更新
            update_metrics(outputs, current_data[self.label_position].to(self.device), statistics, target_variate)
            if self.args.do_predict:
                # 予測結果を保存
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                predictions.append(outputs.detach().cpu().numpy())

        # 全サンプルの性能指標を集計
        metrics = calculate_metrics(statistics)
        mse, mae = metrics['MSE'], metrics['MAE']
        if phase == 'test':
            print('mse:{}, mae:{}'.format(mse, mae))
        if self.args.do_predict:
            return mse, mae, online_data, predictions
        else:
            return mse, mae, online_data

    def analysis_online(self):
        """
        オンライン学習の推論・更新時間計測用
        - 推論時間と更新時間を計測
        - GPUメモリ使用量も確認
        - モデルの計算量（FLOPs）も測定
        """
        # オンライン学習の推論・更新時間計測用
        online_data = get_dataset(self.args, 'test', self.device, wrap_class=self.args.wrap_data_class + [Dataset_Recent],
                                  **self.wrap_data_kwargs)
        # DataLoaderを取得
        online_loader = get_dataloader(online_data, self.args, flag='online')
        # オプティマイザー・損失関数・AMPスケーラーを準備
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        times_update = []  # 更新時間の記録リスト
        times_infer = []   # 推論時間の記録リスト
        print('GPU Mem:', torch.cuda.max_memory_allocated())
        # オンライン学習のメインループ
        for i, (recent_data, current_data) in enumerate(online_loader):
            start_time = time.time()
            self.model.train()  # モデルを訓練モードに
            # recent_dataをデバイスに転送
            recent_data = [d.to(self.device) for d in recent_data]
            # オンライン更新（パラメータ更新）
            loss, _ = self._update_online(recent_data, criterion, model_optim, scaler)
            if i > 10:
                # 10イテレーション目以降の更新時間を記録
                times_update.append(time.time() - start_time)
            self.model.eval()  # モデルを推論モードに
            with torch.no_grad():
                start_time = time.time()
                # current_dataをデバイスに転送
                current_data = [d.to(self.device) for d in current_data]
                # 推論のみ実行
                self.forward(current_data)
            if i > 10:
                # 10イテレーション目以降の推論時間を記録
                times_infer.append(time.time() - start_time)
            if i == 50:
                # 50イテレーションで打ち切り
                break
        print('Final GPU Mem:', torch.cuda.max_memory_allocated())
        # 最小・最大値を除いた平均更新時間・推論時間を計算
        times_update = (sum(times_update) - min(times_update) - max(times_update)) / (len(times_update) - 2)
        times_infer = (sum(times_infer) - min(times_infer) - max(times_infer)) / (len(times_infer) - 2)
        print('Update Time:', times_update)
        print('Infer Time:', times_infer)
        print('Latency:', times_update + times_infer)
        # モデルのFLOPs（計算量）を測定
        test_params_flop(self.model, (1, self.args.seq_len, self.args.enc_in))




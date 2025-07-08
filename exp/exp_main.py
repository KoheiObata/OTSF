"""
メイン実験クラス
基本的な時系列予測実験を実行するクラス
Exp_Basicを継承して具体的な実験処理を実装
"""

import importlib
import os
import time
import warnings
import numpy as np

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

import models.normalization #RevIn
import settings
from data_provider.data_factory import get_dataset
from exp.exp_basic import Exp_Basic
from util.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop, load_model_compile
from util.metrics import metric, update_metrics, calculate_metrics


warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _unfreeze(self, model):
        pass

    @property
    def _model(self):
        if self.args.local_rank >= 0:
            return self.model.module
        return self.model

    def _build_model(self, model=None, framework_class=None):
        """
        モデルを構築するメソッド
        Exp_Basicで定義された抽象メソッドの具体的な実装

        Args:
            model: 既存のモデル（Noneの場合は新規作成）
            framework_class: モデルをラップするフレームワーククラス（オプション）

        Returns:
            torch.nn.Module: 構築されたモデル
        """
        # =====================================
        # 1. 基本モデルの作成
        # =====================================
        if model is None:
            if self.args.model.endswith('_Ensemble'):
                # Ensembleモデルの場合：複数のモデルを組み合わせたアンサンブルモデルを作成
                # 例：'PatchTST_Ensemble' -> models.PatchTST.Model_Ensemble
                base_model_name = self.args.model[:-len('_Ensemble')]
                model = importlib.import_module(f'models.{base_model_name}').Model_Ensemble(
                    self.args).float()
            else:
                # 通常のモデルの場合：単一モデルを作成
                # 例：'PatchTST' -> models.PatchTST.Model
                # モジュールを動的にインポート
                model = importlib.import_module(f'models.{self.args.model}').Model(self.args).float()

        # =====================================
        # 2. 正規化層の追加
        # =====================================
        if self.args.normalization and self.args.online_method != 'OneNet' and self.args.model != 'FSNet_Ensemble':
            # 正規化が必要で、OneNetやFSNet_Ensembleでない場合
            # ForecastModelでモデルをラップし、RevINなどの正規化処理を追加
            model = models.normalization.ForecastModel(
                model,
                num_features=self.args.enc_in,  # 入力特徴量数
                seq_len=self.args.seq_len,      # シーケンス長
                process_method=self.args.normalization  # 正規化手法（例：'revin'）
            )

        # =====================================
        # 3. チェックポイントからの読み込み
        # =====================================
        if hasattr(self.args, 'load_path'):
            # modelを更新する場合freeze=Falseになる(基本的にこの設定)
            if not self.args.freeze:
                # パラメータが凍結されていない場合、最適化器も再作成
                self.model_optim = self._select_optimizer(model=model.to(self.device))
            print('Load checkpoints from', self.args.load_path)
            # チェックポイントからモデルパラメータを読み込み
            model = self.load_checkpoint(self.args.load_path, model)
            if self.model_optim is not None:
                print('Learning rate of model_optim is', self.model_optim.param_groups[0]['lr'])

            if self.args.freeze:
                # パラメータを凍結（勾配計算を無効化）
                model.requires_grad_(False)

        # =====================================
        # 4. フレームワーククラスによるラップ
        # =====================================
        # modelを外側からラップする
        model_params = sum([param.nelement() for param in model.parameters()])
        # ProceedとOneNetではframework_classが指定される
        if framework_class is not None:
            if isinstance(framework_class, list):
                # 複数のフレームワーククラスがある場合、順次適用
                for cls in framework_class:
                    model = cls(model, self.args)
            else:
                # 単一のフレームワーククラスを適用
                model = framework_class(model, self.args)

            # パラメータ数の変化を記録
            new_model_params = sum([param.nelement() for param in model.parameters()])
            print(f'Number of Params: {model_params} -> {new_model_params} (+{new_model_params - model_params})')
            self.model_params = model_params

            # 最適化器に新しいパラメータを追加
            if self.model_optim is not None:
                param_set = set()
                for group in self.model_optim.param_groups:
                    param_set.update(set(group['params']))
                # 既存の最適化器に含まれていない新しいパラメータを検索
                new_params = list(filter(lambda p: p not in param_set and p.requires_grad, model.parameters()))
                if len(new_params) > 0:
                    # 新しいパラメータグループを最適化器に追加
                    self.model_optim.add_param_group({'params': new_params})

        # =====================================
        # 5. 分散学習の設定
        # =====================================
        if self.args.use_multi_gpu and self.args.use_gpu:
            # マルチGPU（DataParallel）の設定
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        elif self.args.local_rank != -1:
            # 分散学習（DistributedDataParallel）の設定
            model = model.to(self.device)
            model = DDP(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=self.args.find_unused_parameters
            )

        # =====================================
        # 6. PyTorch 2.0のコンパイル（オプション）
        # =====================================
        if torch.__version__ >= '2' and self.args.compile:
            print('Compile the model by Pytorch 2.0')
            model = torch.compile(model)

        return model

    def _process_batch(self, batch):
        batch = super()._process_batch(batch)
        batch_x, batch_y = batch[:2]
        if self.args.model in settings.need_x_y_mark:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:4]

            # decoder input
            dec_inp = torch.zeros_like(batch_x[:, -self.args.pred_len:, :])
            dec_inp = torch.cat([batch_x[:, -self.args.label_len:, :], dec_inp], dim=1)

            inp = [batch_x, batch_x_mark, dec_inp, batch_y_mark] + batch[4:]
        elif self.args.model in settings.need_x_mark or hasattr(self.args, 'online_method') and self.args.online_method == 'OneNet':
            # batch=[batch_x, batch_x_mark]にする
            batch = batch[:3] + batch[4:] #batch_y_markを削除
            inp = [batch_x] + batch[2:] #batch_yを削除
        else:
            # batch=[batch_x]にする
            batch = batch[:2] + batch[4:]
            inp = [batch_x] + batch[2:]
        return inp

    def vali(self, vali_data, vali_loader, criterion):
        """
        検証（バリデーション）処理を実行するメソッド

        Args:
            vali_data: 検証データセット
            vali_loader: 検証データローダー
            criterion: 損失関数

        Returns:
            float: 平均検証損失
        """
        self.phase = 'val'  # フェーズを検証に設定
        total_loss = []
        self.model.eval()  # モデルを評価モードに設定

        # 勾配計算を無効化してメモリ使用量を削減
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                # 順伝播で予測を取得
                outputs = self.forward(batch)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # タプルの場合は最初の要素を使用

                # 正解値を取得
                true = batch[self.label_position]
                if not self.args.pin_gpu:
                    true = true.to(self.device)

                # 損失を計算
                loss = criterion(outputs, true)
                total_loss.append(loss.item())

        # 平均損失を計算
        total_loss = np.average(total_loss)

        # 分散学習の場合、全プロセス間で損失を同期
        if self.args.local_rank != -1:
            total_loss = torch.tensor(total_loss, device=self.device)
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item()

        return total_loss

    def train(self, setting, train_data=None, train_loader=None, vali_data=None, vali_loader=None):
        """
        モデルの訓練処理を実行するメソッド

        Args:
            setting: 実験設定名
            train_data: 訓練データセット（Noneの場合は自動取得）
            train_loader: 訓練データローダー（Noneの場合は自動取得）
            vali_data: 検証データセット（Noneの場合は自動取得）
            vali_loader: 検証データローダー（Noneの場合は自動取得）

        Returns:
            tuple: (モデル, 訓練データ, 訓練ローダー, 検証データ, 検証ローダー)
        """
        # =====================================
        # 1. データの準備
        # =====================================
        if train_data is None:
            train_data, train_loader = self._get_data(flag='train')
        if vali_data is None and not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')

        # =====================================
        # 2. チェックポイントパスの設定
        # =====================================
        if self.args.checkpoints:
            path = os.path.join(self.args.checkpoints, setting)
        else:
            path = None

        # =====================================
        # 3. 訓練の初期設定
        # =====================================
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()  # 最適化器の選択
        criterion = self._select_criterion()    # 損失関数の選択

        # 自動混合精度の設定
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        # =====================================
        # 4. 学習率スケジューラーの設定
        # =====================================
        if self.args.lradj == 'TST':
            # TST用のOneCycleLRスケジューラー
            scheduler = lr_scheduler.OneCycleLR(
                optimizer=model_optim,
                steps_per_epoch=train_steps,
                pct_start=self.args.pct_start,
                epochs=self.args.train_epochs,
                max_lr=self.args.learning_rate
            )
        elif self.args.model == 'GPT4TS':
            # GPT4TS用のCosineAnnealingLRスケジューラー
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                model_optim, T_max=self.args.tmax, eta_min=1e-8
            )
        else:
            scheduler = None

        # =====================================
        # 5. エポックループ
        # =====================================
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            # 分散学習時のサンプラー設定
            if self.args.local_rank != -1:
                train_loader.sampler.set_epoch(epoch)
                if hasattr(self, 'online_phases') and 'val' not in self.online_phases:
                    vali_loader.sampler.set_epoch(epoch)

            self.model.train()  # モデルを訓練モードに設定
            epoch_time = time.time()

            # =====================================
            # 6. バッチループ（1エポック分の訓練）
            # =====================================
            for i, batch in enumerate(train_loader):
                self.phase = 'train'
                iter_count += 1

                # モデルの更新（順伝播 + 逆伝播 + パラメータ更新）
                loss, _ = self._update(batch, criterion, model_optim, scaler)
                train_loss.append(loss.item())

                # TSTスケジューラーの場合、ステップごとに学習率を更新
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            # =====================================
            # 7. エポック終了時の処理
            # =====================================
            self.phase = 'train'
            train_loss = np.average(train_loss)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss), end=' ')

            # 検証処理（train_onlyでない場合）
            if not self.args.train_only:
                if epoch >= self.args.begin_valid_epoch:
                    vali_loss = self.vali(vali_data, vali_loader, criterion)
                    print("Vali Loss: {:.7f}".format(vali_loss))
                    early_stopping(vali_loss, self, path)
                else:
                    print()
            else:
                # train_onlyの場合、訓練損失でearly stopping
                early_stopping(train_loss, self, path)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            # Early stoppingのチェック
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # 学習率の調整（TST以外の場合）
            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        # =====================================
        # 8. 訓練終了後の処理
        # =====================================
        if self.args.train_epochs > 0:
            print('Best Valid MSE:', -early_stopping.best_score)
            # 最良のチェックポイントをロード
            self.load_state_dict(early_stopping.best_checkpoint,
                                 strict=not (hasattr(self.args, 'freeze') and self.args.freeze))

            # チェックポイントの保存
            if path and self.args.local_rank <= 0:
                if not os.path.exists(path):
                    os.makedirs(path)
                print('Save checkpoint to', path)
                torch.save(self.state_dict(local_rank=self.args.local_rank), path + '/' + 'checkpoint.pth')

        return self.model, train_data, train_loader, vali_data, vali_loader

    def test(self, setting, test_data=None, test_loader=None, test=0, target_variate=None):
        """
        テスト処理を実行するメソッド

        Args:
            setting: 実験設定名
            test_data: テストデータセット（Noneの場合は自動取得）
            test_loader: テストデータローダー（Noneの場合は自動取得）
            test: テストフラグ（0以外の場合、チェックポイントをロード）
            target_variate: ターゲット変数（特定の変数のみ評価する場合）

        Returns:
            tuple: (MSE, MAE, テストデータ, テストローダー)
        """
        self.phase = 'test'

        # テストデータの準備
        if test_data is None:
            test_data, test_loader = self._get_data(flag='test')

        # テストフラグが設定されている場合、チェックポイントをロード
        if test:
            path = os.path.join("checkpoints", setting, 'checkpoint.pth')
            print('Loading', path)
            self.load_checkpoint(path)

        # =====================================
        # テスト実行
        # =====================================
        self.model.eval()  # モデルを評価モードに設定

        # 統計情報の初期化
        statistics = {k: 0 for k in ['total', 'y_sum', 'MSE', 'MAE']}

        # 勾配計算を無効化してメモリ使用量を削減
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                # 順伝播で予測を取得
                outputs = self.forward(batch)

                # 正解値を取得
                true = batch[self.label_position]
                if not self.args.pin_gpu:
                    true = true.to(self.device)

                # メトリクスの更新
                update_metrics(outputs, true, statistics, target_variate)

        # 最終的なメトリクスを計算
        metrics = calculate_metrics(statistics)
        mse, mae = metrics['MSE'], metrics['MAE']
        print('mse:{}, mae:{}'.format(mse, mae))

        return mse, mae, test_data, test_loader

    def predict(self, path, setting, load=False):
        """
        予測処理を実行するメソッド（未来データの予測）

        Args:
            path: チェックポイントファイルのパス
            setting: 実験設定名
            load: チェックポイントをロードするかどうか
        """
        # チェックポイントのロード
        if load:
            print('Loading', path)
            self.load_checkpoint(path)

        preds = []
        self.model.eval()  # モデルを評価モードに設定

        # =====================================
        # 予測用データの準備
        # =====================================
        # 境界設定を調整（未来予測用）
        self.args.borders[1][0] = self.args.borders[-1][1]
        self.wrap_data_kwargs['borders'] = self.args.borders

        # 予測用データセットの作成（訓練データ全体を使用）
        data_set = get_dataset(self.args, 'train', self.device,
                               wrap_class=self.args.wrap_data_class, **self.wrap_data_kwargs)

        # 予測用データローダーの作成
        dataloader = DataLoader(
            data_set,
            batch_size=self.args.batch_size,
            shuffle=False,  # 予測時はシャッフルしない
            num_workers=self.args.num_workers,
            drop_last=False,
            pin_memory=False,
        )

        # =====================================
        # 予測実行
        # =====================================
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                # 順伝播で予測を取得
                outputs = self.forward(batch)
                pred = outputs.detach().cpu().numpy()  # CPUに移動してNumPy配列に変換
                preds.append(pred)

        # 予測結果を結合
        preds = np.vstack(preds)

        # 予測結果をファイルに保存
        np.save('./results/' + setting + '_pred.npy', preds)
        return

    def analysis(self):
        """
        モデルの分析処理を実行するメソッド
        推論時間、メモリ使用量、FLOPs（浮動小数点演算数）を測定
        """
        # テストデータセットの取得
        data = get_dataset(self.args, 'test', self.device, wrap_class=self.args.wrap_data_class,
                                  **self.wrap_data_kwargs)

        times_infer = []  # 推論時間のリスト
        print('GPU Mem:', torch.cuda.max_memory_allocated())  # 初期GPUメモリ使用量

        self.model.eval()  # モデルを評価モードに設定

        # =====================================
        # 推論時間の測定
        # =====================================
        with torch.no_grad():
            for i in range(50):  # 50回の推論を実行
                start_time = time.time()

                # データをバッチ形式に変換してデバイスに移動
                current_data = [d.unsqueeze(0).to(self.device) for d in data[i]]

                # 順伝播実行
                self.forward(current_data)

                # 最初の10回は除外（ウォームアップ）
                if i > 10:
                    times_infer.append(time.time() - start_time)

                # 30回目で終了
                if i == 30:
                    break

        # =====================================
        # 結果の出力
        # =====================================
        print('Final GPU Mem:', torch.cuda.max_memory_allocated())  # 最終GPUメモリ使用量

        # 推論時間の平均を計算（最大値と最小値を除外）
        times_infer = (sum(times_infer) - min(times_infer) - max(times_infer)) / (len(times_infer) - 2)
        print('Latency:', times_infer)  # 平均推論時間

        # FLOPs（浮動小数点演算数）の測定
        test_params_flop(self.model, (1, self.args.seq_len, self.args.enc_in))

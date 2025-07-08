import torch
import torch.nn.functional as F
import os
import warnings
from tqdm import tqdm

from util.buffer import Buffer
from exp import Exp_Online
from data_provider.data_factory import data_provider, get_dataset, get_dataloader
from data_provider.data_loader import Dataset_Recent
from util.metrics import metric, update_metrics, calculate_metrics

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


class Exp_SOLIDpp(Exp_Online):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer_size = self.args.buffer_size
        self.threshold = self.args.lambda_period
        self.mini_batch = self.args.mini_batch
        self.buffer = Buffer(self.buffer_size, self.device, sample_selection_strategy='fifo')
        self.count = 0
        self.period = get_period(self.args.dataset)


    def calculate_pairwise_similarity(self, seq_y_current, buff_y):
        """
        ペアワイズ距離を使用して類似度を計算する関数
        """

        # seq_y_currentのshapeを[1, pred_len, features]または[pred_len, features]に揃える
        if seq_y_current.dim() == 4 and seq_y_current.size(0) == 1:
            seq_y_current = seq_y_current.squeeze(0)
        if seq_y_current.dim() == 3 and seq_y_current.size(0) == 1:
            seq_y_current = seq_y_current.squeeze(0)

        # seq_y_currentをバッファサイズ分に拡張
        seq_y_expanded = seq_y_current.unsqueeze(0).expand(buff_y.size(0), -1, -1)

        # 時間次元と特徴次元を平坦化
        seq_y_flat = seq_y_expanded.view(seq_y_expanded.size(0), -1)  # [buffer_size, pred_len * features]
        buff_y_flat = buff_y.view(buff_y.size(0), -1)  # [buffer_size, pred_len * features]

        # F.pairwise_distanceを使用してユークリッド距離を計算
        # F.pairwise_distanceは2つのテンソル間の距離を計算
        # seq_y_flat: [buffer_size, features]
        # buff_y_flat: [buffer_size, features]
        # 結果: [buffer_size] - 各ペア間の距離
        # 距離が小さいほど類似度が高い
        distances = F.pairwise_distance(seq_y_flat, buff_y_flat, p=2)

        # 値が大きいほど類似度が高くする
        similarities = -distances
        return similarities

    def calculate_mse_similarity(self, seq_y_current, buff_y):
        """
        MSE（Mean Squared Error）を使用して類似度を計算する関数
        """
        # seq_y_currentのshapeを[1, pred_len, features]または[pred_len, features]に揃える
        if seq_y_current.dim() == 4 and seq_y_current.size(0) == 1:
            seq_y_current = seq_y_current.squeeze(0)
        if seq_y_current.dim() == 3 and seq_y_current.size(0) == 1:
            seq_y_current = seq_y_current.squeeze(0)
        # seq_y_currentをバッファサイズ分に拡張
        seq_y_expanded = seq_y_current.unsqueeze(0).expand(buff_y.size(0), -1, -1)
        mse = torch.mean((seq_y_expanded - buff_y) ** 2, dim=(1, 2))
        similarities = -mse
        return similarities

    def calculate_dtw_similarity(self, seq_y_current, buff_y):
        """
        DTW（Dynamic Time Warping）を使用して類似度を計算する関数
        """
        def dtw_distance(seq1, seq2):
            n, m = seq1.size(0), seq2.size(0)
            dtw_matrix = torch.full((n + 1, m + 1), float('inf'), device=seq1.device)
            dtw_matrix[0, 0] = 0
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = torch.mean((seq1[i-1] - seq2[j-1]) ** 2)
                    dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
            return dtw_matrix[n, m]
        # seq_y_currentのshapeを[1, pred_len, features]または[pred_len, features]に揃える
        if seq_y_current.dim() == 4 and seq_y_current.size(0) == 1:
            seq_y_current = seq_y_current.squeeze(0)
        if seq_y_current.dim() == 3 and seq_y_current.size(0) == 1:
            seq_y_current = seq_y_current.squeeze(0)
        similarities = torch.zeros(buff_y.size(0), device=seq_y_current.device)
        for i in range(buff_y.size(0)):
            dtw_dist = dtw_distance(seq_y_current, buff_y[i])
            similarities[i] = -dtw_dist
        return similarities

    def find_most_similar_buffer(self, seq_y_current, buff_y, similarity_type='mse'):
        """
        seq_y_currentに最も似ているbuff_yを返す関数
        similarity_type: 'mse', 'dtw', 'vae'（コサイン類似度）
        """
        if similarity_type.lower() == 'mse':
            similarities = self.calculate_mse_similarity(seq_y_current, buff_y)
        elif similarity_type.lower() == 'dtw':
            similarities = self.calculate_dtw_similarity(seq_y_current, buff_y)
        elif similarity_type.lower() == 'pairwise':
            similarities = self.calculate_pairwise_similarity(seq_y_current, buff_y)
        elif similarity_type.lower() == 'random':
            similarities = torch.rand(buff_y.size(0))
        else:
            raise ValueError(f"Unsupported similarity type: {similarity_type}. Use 'mse', 'dtw', or 'vae'.")

        # 最も類似度が高い（値が最大の）インデックスを取得
        most_similar_idx = torch.argmax(similarities)
        most_similar_y = buff_y[most_similar_idx]

        return most_similar_idx, most_similar_y, similarities


    def phase_difference(self, idx, indices):
        """
        インデックス間の位相差を計算する関数

        Args:
            idx: 現在のインデックス（スカラーまたはテンソル）
            indices: 比較対象のインデックス（テンソル）

        Returns:
            similarities: 位相差に基づく類似度（値が小さいほど類似）
        """
        # idxとindicesをテンソルに変換してから剰余演算を実行
        if not isinstance(idx, torch.Tensor):
            idx = torch.tensor(idx, device=indices.device, dtype=indices.dtype)
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, device=idx.device, dtype=idx.dtype)

        mod_idx = idx % self.period  # mod()の代わりに%演算子を使用
        mod_indices = indices % self.period

        # mod_idxをmod_indicesと同じshapeにする
        if mod_idx.dim() == 0:  # スカラーの場合
            mod_idx = mod_idx.expand_as(mod_indices)
        elif mod_idx.dim() == 1 and mod_indices.dim() > 1:  # 1次元の場合
            mod_idx = mod_idx.unsqueeze(0).expand_as(mod_indices)

        phase_diff = abs(mod_idx - mod_indices) / self.period

        return phase_diff

    def get_adjust_data(self, batch, current_batch=None):
        seq_x, seq_y, seq_x_mark, seq_y_mark = batch
        if current_batch is None:
            current_batch = batch
        seq_x_current, seq_y_current, seq_x_mark_current, seq_y_mark_current = current_batch

        # bufferを作成
        buff = self.buffer.get_data(self.buffer.buffer_size) #[1,60,14].[1,1,7]
        buff_x, buff_y, buff_x_mark, buff_y_mark, index = buff

        # recent_batchのindex
        recent_idx = self.count + 1
        current_idx = recent_idx + self.args.pred_len - 1

        # 位相差を計算
        phase_diff = self.phase_difference(current_idx, index)

        # 位相差が閾値より小さい場合のみ選択（1: 選択, 0: 除外）
        mask = (phase_diff < self.threshold).float()

        # マスクに基づいてバッファデータを選択
        selected_indices = torch.where(mask == 1)[0]

        if len(selected_indices) == 0:
            # 選択されたデータがない場合は元のデータを使用
            selected_buff_x = buff_x
            selected_buff_y = buff_y
            selected_buff_x_mark = buff_x_mark
            selected_buff_y_mark = buff_y_mark
        else:
            selected_buff_x = buff_x[selected_indices]
            selected_buff_y = buff_y[selected_indices]
            selected_buff_x_mark = buff_x_mark[selected_indices]
            selected_buff_y_mark = buff_y_mark[selected_indices]

        # 類似度を計算
        most_similar_idx, most_similar_x, similarities = self.find_most_similar_buffer(seq_x_current, selected_buff_x, similarity_type='pairwise')

        # 類似度が最大のものをmini_batch個選択
        out, inds = torch.topk(similarities, min(self.args.mini_batch, len(similarities)))

        topk_buff_x = selected_buff_x[inds]
        topk_buff_y = selected_buff_y[inds]
        topk_buff_x_mark = selected_buff_x_mark[inds]
        topk_buff_y_mark = selected_buff_y_mark[inds]

        augmented_data = [topk_buff_x, topk_buff_y, topk_buff_x_mark, topk_buff_y_mark]
        return augmented_data

    def train_loss(self, criterion, batch, outputs):
        loss = super().train_loss(criterion, batch, outputs)
        return loss

    def _update(self, batch, criterion, optimizer, scaler=None, current_batch=None):
        if not isinstance(optimizer, tuple):
            optimizer = (optimizer, )
        for optim in optimizer:
            optim.zero_grad()
        if self.buffer.len()>self.args.buffer_size:
            batch=self.get_adjust_data(batch, current_batch=current_batch)
            # pass
        outputs = self.forward(batch)
        loss = self.train_loss(criterion, batch, outputs)
        # if self.buffer.len()>100:
            # print('loss_aug', loss_aug, 'loss_recent', loss_recent, 'aug_ratio', aug_ratio)
        if self.args.use_amp:
            scaler.scale(loss).backward()
            for optim in optimizer:
                scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            for optim in optimizer:
                optim.step()
        return loss, outputs


    def _update_online(self, batch, criterion, optimizer, scaler=None, current_batch=None):
        loss, outputs = self._update(batch, criterion, optimizer, scaler=None, current_batch=current_batch)
        idx = self.count + torch.arange(batch[1].size(0)).to(self.device)
        self.count += batch[1].size(0)
        self.buffer.add_data(*batch, idx)
        return loss, outputs

    def update_valid(self, valid_data=None):
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
        valid_loader = get_dataloader(valid_data, self.args, 'online')

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None
        predictions = []
        for i, (recent_batch, current_batch) in enumerate(tqdm(valid_loader, mininterval=10)):
            self.model.train()
            self._update_online(recent_batch, criterion, model_optim, scaler, current_batch=current_batch)
            if self.args.do_predict:
                self.model.eval()
                with torch.no_grad():
                    outputs = self.forward(current_batch)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                predictions.append(outputs.detach().cpu().numpy())
        return predictions

    def online(self, online_data=None, target_variate=None, phase='test', show_progress=False):
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
        if online_data is None:
            online_data = get_dataset(self.args, phase, self.device, wrap_class=self.args.wrap_data_class + [Dataset_Recent], **self.wrap_data_kwargs)
        online_loader = get_dataloader(online_data, self.args, flag='online')

        if self.args.do_predict:
            predictions = []
        statistics = {k: 0 for k in ['total', 'y_sum', 'MSE', 'MAE']}

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        if show_progress:
            online_loader = tqdm(online_loader, mininterval=10)

        for i, (recent_data, current_data) in enumerate(online_loader):
            self.model.train()
            self._update_online(recent_data, criterion, model_optim, scaler, current_batch=current_data)
            self.model.eval()
            with torch.no_grad():
                outputs = self.forward(current_data)
                if self.args.do_predict:
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    predictions.append(outputs.detach().cpu().numpy())
                update_metrics(outputs, current_data[self.label_position].to(self.device), statistics, target_variate)

        metrics = calculate_metrics(statistics)
        mse, mae = metrics['MSE'], metrics['MAE']
        if phase == 'test':
            print('mse:{}, mae:{}'.format(mse, mae))
        if self.args.do_predict:
            return mse, mae, online_data, predictions
        else:
            return mse, mae, online_data

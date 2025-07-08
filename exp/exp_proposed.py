import torch
import os
import warnings
from tqdm import tqdm

from util.buffer import Buffer
from exp import Exp_Online
from data_provider.data_factory import data_provider, get_dataset, get_dataloader
from data_provider.data_loader import Dataset_Recent
from util.metrics import metric, update_metrics, calculate_metrics

from tool_btoa.vae_quant import setup_the_VAE, VAE, train_VAE
from tool_btoa.new_augmentations import *
from tool_proposed import augmentations

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


class Exp_Proposed(Exp_Online):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer_size = self.args.buffer_size
        self.buffer = Buffer(self.buffer_size, self.device)
        self.count = 0
        self.period = get_period(self.args.dataset)

        # self.vae = self.load_vae()

    def calculate_similarity_latents(self, sample):
        qz_params = self.vae.encoder.forward(sample.to(self.device).float()).view(sample.size(0), self.args.latent_dim, self.vae.q_dist.nparams).data
        latent_values = self.vae.q_dist.sample(params=qz_params)
        a_norm = latent_values / latent_values.norm(dim=1)[:, None]
        b_norm = latent_values / latent_values.norm(dim=1)[:, None]
        res = torch.mm(a_norm, b_norm.transpose(0,1))
        res = res.fill_diagonal_(0) # Make diagonals to 0
        return res

    def calculate_vae_similarity(self, seq_y_current, buff_y):
        # VAEのエンコーダで潜在ベクトルに変換
        # seq_y_current: [1, pred_len, features] or [pred_len, features]
        if seq_y_current.dim() == 4 and seq_y_current.size(0) == 1:
            seq_y_current = seq_y_current.squeeze(0)
        if seq_y_current.dim() == 3 and seq_y_current.size(0) == 1:
            seq_y_current = seq_y_current.squeeze(0)
        if seq_y_current.dim() == 2:
            seq_y_current = seq_y_current.unsqueeze(0)
        # buff_y: [buffer_size, pred_len, features]
        all_y = torch.cat([seq_y_current, buff_y], dim=0)  # [1+buffer_size, pred_len, features]
        qz_params = self.vae.encoder.forward(all_y.to(self.device).float()).view(all_y.size(0), self.args.latent_dim, self.vae.q_dist.nparams).data
        latent_values = self.vae.q_dist.sample(params=qz_params)  # [1+buffer_size, latent_dim]
        # コサイン類似度計算
        query = latent_values[0]  # [latent_dim]
        keys = latent_values[1:]  # [buffer_size, latent_dim]
        query_norm = query / query.norm()
        keys_norm = keys / keys.norm(dim=1, keepdim=True)
        similarities = torch.matmul(keys_norm, query_norm)  # [buffer_size]
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
        elif similarity_type.lower() == 'vae':
            similarities = self.calculate_vae_similarity(seq_y_current, buff_y)
        elif similarity_type.lower() == 'random':
            similarities = torch.rand(buff_y.size(0))
        else:
            raise ValueError(f"Unsupported similarity type: {similarity_type}. Use 'mse', 'dtw', or 'vae'.")

        # 最も類似度が高い（値が最大の）インデックスを取得
        most_similar_idx = torch.argmax(similarities)
        most_similar_y = buff_y[most_similar_idx]

        return most_similar_idx, most_similar_y, similarities

    def load_vae(self):
        prior_dist, q_dist = setup_the_VAE(self.args)
        vae = VAE(z_dim=self.args.latent_dim, args=self.args, use_cuda=True, prior_dist=prior_dist, q_dist=q_dist, include_mutinfo=not self.args.exclude_mutinfo, tcvae=self.args.tcvae, mss=self.args.mss).to(self.device)
        if not os.path.isfile(self.args.save+'/checkpt-0000.pth'):
            train_data, train_loader = self._get_data('train')
            vae_model = train_VAE(train_loader, self.args, self.device)
        vae_model = torch.load(self.args.save+'/checkpt-0000.pth')
        vae.load_state_dict(vae_model['state_dict'])
        vae.eval()
        return vae

    def get_adjust_data_btoa(self,batch, current_batch=None):
        seq_x, seq_y, seq_x_mark, seq_y_mark = batch

        # bufferを作成（500個）
        buff = self.buffer.get_data(self.buffer.buffer_size) #[1,60,14].[1,1,7]
        buff_x, buff_y, buff_x_mark, buff_y_mark, index = buff

        # VAEでencodeして類似度を計算
        most_similar_idx, most_similar_x, similarities = self.find_most_similar_buffer(seq_x, buff_x, similarity_type='vae')

        # 類似度が最大のものをmini_batch個(15個)選択
        out, inds = torch.topk(similarities,self.args.mini_batch)

        topk_buff_x = buff_x[inds]
        topk_similarities = similarities[inds]

        # BTOA(フーリエ変換)
        seq_x_aug = augmentations.fft_mix(seq_x, topk_buff_x, topk_similarities, self.args).to(self.device)
        seq_x_aug = torch.concatenate((seq_x, seq_x_aug),dim=0)

        seq_y_aug = seq_y.repeat(self.args.mini_batch+1, 1, 1).to(self.device)
        seq_x_mark_aug = seq_x_mark.repeat(self.args.mini_batch+1, 1, 1).to(self.device)
        seq_y_mark_aug = seq_y_mark.repeat(self.args.mini_batch+1, 1, 1).to(self.device)

        augmented_data = [seq_x_aug, seq_y_aug, seq_x_mark_aug, seq_y_mark_aug]
        return augmented_data

    def get_adjust_data_2(self, batch, current_batch=None):
        seq_x, seq_y, seq_x_mark, seq_y_mark = batch
        if current_batch is None:
            current_batch = batch
        seq_x_current, seq_y_current, seq_x_mark_current, seq_y_mark_current = current_batch

        # bufferを作成
        buff = self.buffer.get_data(self.args.buffer_size)
        buff_x, buff_y, buff_x_mark, buff_y_mark, index = buff

        # recent_batchのindex
        recent_idx = self.count + 1
        current_idx = recent_idx + self.args.pred_len - 1

        # bufferからランダムに選択
        most_similar_idx, most_similar_x, similarities = self.find_most_similar_buffer(seq_x_current, buff_x, similarity_type='random')

        # 類似度が最大のものをmini_batch個(15個)選択
        out, inds = torch.topk(similarities,self.args.mini_batch)

        topk_buff_x = buff_x[inds]
        topk_buff_y = buff_y[inds]
        topk_index = index[inds]


        # topk_buff_x = torch.cat((topk_buff_x, seq_x), dim=0)
        # topk_buff_y = torch.cat((topk_buff_y, seq_y), dim=0)

        # 0~self.buffer.buffer_sizeの中からindに含まれない数字をランダムに一つ選択
        # random_idx = torch.randint(0, self.buffer.buffer_size-1, (1,))
        # while random_idx in inds:
            # random_idx = torch.randint(0, self.buffer.buffer_size-1, (1,))
        # seq_x_random = buff_x[random_idx].clone().detach()


        # そのまま (めっちゃ悪くなる)
        # aug_sample2 = x_aug[inds]

        # ノイズを足す (めっちゃ悪くなる)
        # aug_sample2 = x_aug + torch.randn_like(x_aug) * 0.1


        # 提案手法(フーリエ変換)
        # aug_sample2 = gen_new_aug_2(seq_x_current, topk_buff_x, topk_similarities, self.args).to(self.device)
        # aug_sample2 = torch.concatenate((seq_x_current, aug_sample2),dim=0)

        # seq_y_aug = gen_new_aug_2(seq_y_current, topk_buff_y, topk_similarities, self.args).to(self.device)
        # seq_y_aug = torch.concatenate((outputs, seq_y_aug),dim=0)

        # seq_y_aug = topk_buff_y
        # aug_sample2 = gen_new_aug(seq_x_current, topk_buff_x, base_ratio_start=0.8, base_ratio_end=0.8, base_seq='x').to(self.device)


        # topk_buff_xy = torch.cat((topk_buff_x, topk_buff_y), dim=1)
        # seq_xy_current = torch.cat((seq_x_current, seq_y_current), dim=1)
        # seq_xy_current = torch.cat((seq_x_current, outputs), dim=1)
        # aug_sample2_xy = gen_new_aug(seq_xy_current, topk_buff_xy, base_ratio_start=0.5, base_ratio_end=0.5, base_seq='x').to(self.device)
        # aug_sample2 = aug_sample2_xy[:, :self.args.seq_len, :]
        # seq_y_aug = aug_sample2_xy[:, self.args.seq_len:, :]

        # aug_sample2 = aug_sample2_xy[:, :self.args.seq_len, :]
        # lags = -((current_idx - topk_index) % self.period) - int(self.period/2)
        lags = -((current_idx - topk_index) % self.period) - self.period
        # lags = -((current_idx - topk_index) % self.period)
        # lagsの最小値は-args.seq_len+1である．最小値より小さい値はself.periodを加える．
        while lags.min() < -self.args.seq_len+1:
            lags = torch.where(lags < -self.args.seq_len+1, lags+self.period, lags)
        # lagsの要素をintに変換
        if isinstance(lags, torch.Tensor):
            lags = lags.int()
        else:
            lags = int(lags)
        # print('lags', lags)
        self.args.detrend = False
        seq_x_aug = augmentations.gen_new_aug(seq_x_current, topk_buff_x, base_ratio_start=0.5, base_ratio_end=0.5, base_seq='x', period=self.period, lags=lags, args=self.args).to(self.device)
        # aug_sample2 = topk_buff_x
        # aug_sample2 = gen_new_aug(seq_x_random, topk_buff_x, base_ratio_start=0.5, base_ratio_end=0.5, base_seq='x').to(self.device)

        # seq_y_aug = seq_y
        # aug_sample2 = gen_new_aug(seq_x_current, seq_x, base_ratio_start=0.5, base_ratio_end=0.5, base_seq='x').to(self.device)

        # y_aug=torch.cat((seq_y,buff_y),dim=0)
        # seq_y_aug =torch.cat((seq_y,y_aug[inds]))
        # seq_y_aug =y_aug[inds]
#

        seq_y_aug = topk_buff_y
        seq_x_mark_aug = seq_x_mark_current.repeat(self.args.mini_batch, 1, 1).to(self.device)
        seq_y_mark_aug = seq_y_mark_current.repeat(self.args.mini_batch, 1, 1).to(self.device)

        augmented_data = [seq_x_aug, seq_y_aug, seq_x_mark_aug, seq_y_mark_aug]

        return augmented_data

    def train_loss(self, criterion, batch, outputs):
        loss = super().train_loss(criterion, batch, outputs)
        return loss

    def _update(self, batch, criterion, optimizer, scaler=None, current_batch=None):
        if not isinstance(optimizer, tuple):
            optimizer = (optimizer, )
        for optim in optimizer:
            optim.zero_grad()
        loss_aug = 0
        if self.buffer.len()>self.args.buffer_size:
            batch_aug=self.get_adjust_data_2(batch, current_batch=current_batch)
            outputs_aug = self.forward(batch_aug)
            loss_aug = self.train_loss(criterion, batch_aug, outputs_aug)
            # pass
        outputs = self.forward(batch)
        loss_recent = self.train_loss(criterion, batch, outputs)
        aug_ratio = 0.5
        loss = loss_aug*aug_ratio + loss_recent*(1-aug_ratio)
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

    def _update_original(self, batch, criterion, optimizer, scaler=None, current_batch=None):
        if not isinstance(optimizer, tuple):
            optimizer = (optimizer, )
        for optim in optimizer:
            optim.zero_grad()
        if self.buffer.len()>self.args.buffer_size:
            batch=self.get_adjust_data(batch, current_batch=current_batch)
        outputs = self.forward(batch)
        loss = self.train_loss(criterion, batch, outputs)
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

        # 元のDataLoaderを保存（tqdmでラップされる前）
        original_loader = online_loader

        if show_progress:
            online_loader = tqdm(online_loader, mininterval=10)

        for i, (recent_data, current_data) in enumerate(online_loader):
            # print('i', i)
            # print('recent_data', recent_data[0][0, :2, 0])
            # print('current_data', current_data[0][0, :2, 0])
            # # __getitem__を使用してサンプルを取り出す（元のDataLoaderを使用）
            # recent_data_i = original_loader.dataset[i]
            # current_data_i = original_loader.dataset[i+self.args.pred_len]
            # print('recent_data_i', recent_data_i[0][0][:2, 0]) recent_dataと一致
            # print('recent_data_i', recent_data_i[1][0][:2, 0]) current_dataと一致
            # print('current_data_i', current_data_i[0][0][:2, 0]) current_dataと一致
            # print('current_data_i', current_data_i[1][0][:2, 0]) current_data+pred_lenと一致
            # exit()
            past_length = 0
            if i+self.args.pred_len-past_length > 0:
                current_data_i = original_loader.dataset[i+self.args.pred_len-past_length]
                self.model.train()
                self._update_online(recent_data, criterion, model_optim, scaler, current_batch=current_data_i[0])
            else:
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

import torch
import os
import warnings

from util.buffer import Buffer
from exp import Exp_Online

from tool_btoa.vae_quant import setup_the_VAE, VAE, train_VAE
from tool_btoa.new_augmentations import *

warnings.filterwarnings('ignore')



class Exp_BTOA(Exp_Online):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer_size = self.args.buffer_size
        self.buffer = Buffer(self.buffer_size, self.device)
        self.count = 0

        self.vae = self.load_vae()

    def calculate_similarity_latents(self, sample):
        qz_params = self.vae.encoder.forward(sample.to(self.device).float()).view(sample.size(0), self.args.latent_dim, self.vae.q_dist.nparams).data
        latent_values = self.vae.q_dist.sample(params=qz_params)
        a_norm = latent_values / latent_values.norm(dim=1)[:, None]
        b_norm = latent_values / latent_values.norm(dim=1)[:, None]
        res = torch.mm(a_norm, b_norm.transpose(0,1))
        res = res.fill_diagonal_(0) # Make diagonals to 0
        return res

    def load_vae(self):
        prior_dist, q_dist = setup_the_VAE(self.args)
        vae = VAE(z_dim=self.args.latent_dim, args=self.args, use_cuda=True, prior_dist=prior_dist, q_dist=q_dist, include_mutinfo=not self.args.exclude_mutinfo, tcvae=self.args.tcvae, mss=self.args.mss).to(self.device)
        if not os.path.isfile(self.args.save+'/checkpt-0000.pth'):
            batch_size = self.args.batch_size
            self.args.batch_size = 2048
            train_data, train_loader = self._get_data('train')
            vae_model = train_VAE(train_loader, self.args, self.device)
            self.args.batch_size = batch_size
        vae_model = torch.load(self.args.save+'/checkpt-0000.pth')
        vae.load_state_dict(vae_model['state_dict'])
        vae.eval()
        return vae

    def get_adjust_data(self,batch):
        seq_x, seq_y, seq_x_mark, seq_y_mark = batch

        # bufferを作成（100個）
        buff = self.buffer.get_data(self.buffer_size) #[1,60,14].[1,1,7]
        buff_x, buff_y, buff_x_mark, buff_y_mark, idx = buff
        # バッファと現在のデータを結合
        x_aug=torch.cat((seq_x,buff_x),dim=0)
        # VAEでencodeして類似度を計算
        similarities = self.calculate_similarity_latents(x_aug)
        # 類似度が最大のものをmini_batch個(15個)選択
        out, inds = torch.topk(similarities[0],self.args.mini_batch)
        # バッファと現在のデータを結合
        x_aug=torch.cat((seq_x,x_aug[inds]))

        # 拡張データを生成
        aug_sample2 = gen_new_aug_2(x_aug, self.args, inds, out, similarities[0]).to(self.device)
        aug_sample2[:1,:,:]=seq_x

        seq_y_aug = seq_y.repeat(self.args.mini_batch+1, 1, 1).to(self.device)
        seq_x_mark_aug = seq_x_mark.repeat(self.args.mini_batch+1, 1, 1).to(self.device)
        seq_y_mark_aug = seq_y_mark.repeat(self.args.mini_batch+1, 1, 1).to(self.device)

        augmented_data = [aug_sample2, seq_y_aug, seq_x_mark_aug, seq_y_mark_aug]

        return augmented_data

    def train_loss(self, criterion, batch, outputs):
        loss = super().train_loss(criterion, batch, outputs)
        return loss

    def _update(self, batch, criterion, optimizer, scaler=None):
        if not isinstance(optimizer, tuple):
            optimizer = (optimizer, )
        for optim in optimizer:
            optim.zero_grad()
        if self.buffer.len()>self.buffer_size:
            batch=self.get_adjust_data(batch)
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

    def _update_online(self, batch, criterion, optimizer, scaler=None):
        loss, outputs = self._update(batch, criterion, optimizer, scaler=None)
        idx = self.count + torch.arange(batch[1].size(0)).to(self.device)
        self.count += batch[1].size(0)
        self.buffer.add_data(*batch, idx)
        return loss, outputs

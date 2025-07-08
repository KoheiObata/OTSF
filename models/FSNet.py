import torch
import torch.nn as nn

from layers.RevIN import RevIN
from layers.ts2vec.encoder import TS2VecEncoderWrapper
from layers.ts2vec.fsnet import TSEncoder

class Model(nn.Module):
    """
    FSNet（Fast and Slow Network）モデル本体
    - TS2Vecベースのエンコーダを用いた時系列特徴抽出
    - 入力: x (B, T, D), x_mark (B, T, time_features)
    - 出力: 予測値 (B, pred_len, c_out)
    - store_grad/try_trigger_はPadConv層の特殊機能
    """
    def __init__(self, args):
        super().__init__()
        # TS2Vecエンコーダの構築
        encoder = TSEncoder(input_dims=args.enc_in + (4 if args.timeenc == 1 else 7),
                            output_dims=320,  # ts2vecの標準値
                            hidden_dims=64,   # ts2vecの標準値
                            depth=10)
        self.encoder = TS2VecEncoderWrapper(encoder, mask='all_true')
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.dim = args.c_out * args.pred_len
        # 予測用の全結合層
        self.regressor = nn.Linear(320, self.dim)

    def forward(self, x, x_mark=None):
        """
        x: 入力系列 (B, T, D)
        x_mark: 時刻特徴量 (B, T, time_features)
        返り値: 予測値 (B, pred_len, c_out)
        """
        if x_mark is None:
            x_mark = torch.zeros(*x.shape[:2], 7, device=x.device)
        x = torch.cat([x, x_mark], dim=-1)
        rep = self.encoder(x)  # 特徴抽出
        y = self.regressor(rep)
        y = y.reshape(len(y), self.pred_len, -1)
        return y

    def store_grad(self):
        """
        PadConv層の勾配保存（FSNetの特殊機能）
        """
        for name, layer in self.encoder.named_modules():
            if 'PadConv' in type(layer).__name__:
                layer.store_grad()

    def try_trigger_(self, flag=True):
        """
        PadConv層のトリガーフラグ切替（FSNetの特殊機能）
        """
        for name, layer in self.encoder.named_modules():
            if 'PadConv' in type(layer).__name__:
                layer.try_trigger = flag


class Model_Ensemble(Model):
    """
    FSNetのアンサンブル拡張モデル
    - 入力系列方向と時刻方向の2つのエンコーダを持つ
    - RevIN正規化（オプション）
    - forward_individual: 2つの系列の個別出力
    - forward: 重み付き和で最終出力
    """
    def __init__(self, args):
        super().__init__(args)
        self.norm = False
        if args.normalization.lower() == 'revin':
            self.norm = True
            self.revin = RevIN(num_features=args.enc_in)
        depth = 10
        # 時刻方向のエンコーダ
        encoder = TSEncoder(input_dims=args.seq_len,
                            output_dims=320,  # ts2vecの標準値
                            hidden_dims=64,   # ts2vecの標準値
                            depth=depth)
        self.encoder_time = TS2VecEncoderWrapper(encoder, mask='all_true')
        self.regressor_time = nn.Linear(320, args.pred_len)

    def forward_individual(self, x, x_mark=None):
        """
        2つの系列方向で個別に予測
        - y1: 時刻方向エンコーダの出力 (B, c_out, pred_len)
        - y2: 通常FSNetの出力 (B, pred_len, c_out)
        """
        rep = self.encoder_time.encoder.forward(x.transpose(1, 2))
        y1 = self.regressor_time(rep).transpose(1, 2)
        if self.norm:
            x = self.revin(x, mode='norm')
        y2 = super().forward(x, x_mark)
        if self.norm:
            y2 = self.revin(y2, mode='denorm')
        return y1, y2

    def forward(self, x, x_mark=None, w1=0.5, w2=0.5):
        """
        2つの系列方向の出力を重み付き和で合成
        - w1, w2: 各系列の重み
        返り値: (最終出力, y1, y2)
        """
        y1, y2 = self.forward_individual(x, x_mark)
        return y1 * w1 + y2 * w2, y1, y2

    def store_grad(self):
        """
        2つのエンコーダのPadConv層の勾配保存
        """
        for name, layer in self.encoder.named_modules():
            if 'PadConv' in type(layer).__name__:
                layer.store_grad()
        for name, layer in self.encoder_time.named_modules():
            if 'PadConv' in type(layer).__name__:
                layer.store_grad()

    def try_trigger_(self, flag=True):
        """
        2つのエンコーダのPadConv層のトリガーフラグ切替
        """
        for name, layer in self.encoder.named_modules():
            if 'PadConv' in type(layer).__name__:
                layer.try_trigger = flag
        for name, layer in self.encoder_time.named_modules():
            if 'PadConv' in type(layer).__name__:
                layer.try_trigger = flag
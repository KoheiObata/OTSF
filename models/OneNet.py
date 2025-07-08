import copy

import torch
from torch import nn

from models import normalization
from models.FSNet import Model as FSNet
from layers.RevIN import RevIN

class OneNet(nn.Module):
    """
    OneNet本体クラス
    - backbone（FSNetなど）を持ち、2系列の出力を重み付きで合成
    - 決定関数MLPで重みを動的に決定可能
    - self.weight: 各変数ごとの重みパラメータ
    - forward: y1, y2（backboneの2系列出力）を合成して返す
    """
    def __init__(self, backbone, args):
        super().__init__()
        self.backbone = backbone
        # 決定関数MLP（重み推定用）
        self.decision = MLP(n_inputs=args.pred_len * 3, n_outputs=1, mlp_width=32, mlp_depth=3, mlp_dropout=0.1, act=nn.Tanh())
        self.weight = nn.Parameter(torch.zeros(args.enc_in))

    def forward(self, *inp):
        """
        inp: (x, x_mark, loss1, loss2) など
        - y1, y2: backboneの2系列出力
        - inp[-2], inp[-1]: 重み（loss1, loss2）
        返り値: (重み付き合成出力, y1, y2)
        """
        flag = False
        if len(inp) == 1:
            inp = inp + (None, 1, 1)
            flag = True
        y1, y2 = self.backbone.forward_individual(*inp[:-2])
        if flag:
            b, t, d = y1.shape
            weight = self.weight.view(1, 1, -1).repeat(b, t, 1)
            loss1 = torch.sigmoid(weight + torch.zeros(b, 1, d, device=weight.device)).view(b, t, d)
            inp = inp[:-2] + (loss1, 1 - loss1)
        return y1.detach() * inp[-2] + y2.detach() * inp[-1], y1, y2

    def store_grad(self):
        self.backbone.store_grad()


class Model_Ensemble(nn.Module):
    """
    OneNet用アンサンブルモデル
    - encoder: FSNetを正規化付きでラップ
    - encoder_time: backbone（通常FSNet）
    - forward_individual: 2系列の個別出力
    - forward: 重み付き和で最終出力
    """
    def __init__(self, backbone, args):
        super().__init__()
        _args = copy.deepcopy(args)
        _args.seq_len = 60
        self.seq_len = 60
        self.encoder = normalization.ForecastModel(FSNet(_args), num_features=args.enc_in, seq_len=60)
        # self.norm = False
        # if args.normalization.lower() == 'revin':
        if args.pretrain and hasattr(args, 'fsnet_path'):
            # if 'RevIN' in args.fsnet_path:
            print('Load FSNet from', args.fsnet_path)
            self.encoder.load_state_dict(torch.load(args.fsnet_path)['model'])
            # else:
            #     self.encoder.load_state_dict(torch.load(args.fsnet_path)['model'])
        self.encoder_time = backbone

    def forward_individual(self, x, x_mark):
        """
        2系列の個別出力
        - y1: encoder_timeの出力
        - y2: encoder(FSNet)の出力
        """
        y1 = self.encoder_time(x, x_mark)
        y2 = self.encoder.forward(x[..., -self.seq_len:, :], x_mark[..., -self.seq_len:, :] if x_mark is not None else None)
        return y1, y2

    def forward(self, x, x_mark, w1=0.5, w2=0.5):
        """
        2系列の出力を重み付き和で合成
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

class MLP(nn.Module):
    """
    汎用多層パーセプトロン（決定関数用）
    - n_inputs: 入力次元
    - n_outputs: 出力次元
    - mlp_width: 隠れ層幅
    - mlp_depth: 層数
    - mlp_dropout: ドロップアウト率
    - act: 活性化関数
    """
    def __init__(self, n_inputs, n_outputs, mlp_width, mlp_depth, mlp_dropout, act=nn.ReLU()):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, mlp_width)
        self.dropout = nn.Dropout(mlp_dropout)
        self.hiddens = nn.ModuleList([
            nn.Linear(mlp_width, mlp_width)
            for _ in range(mlp_depth-2)])
        self.output = nn.Linear(mlp_width, n_outputs)
        self.n_outputs = n_outputs
        self.act = act

    def forward(self, x, train=True):
        """
        x: 入力テンソル
        train: ドロップアウト有効化フラグ
        返り値: 出力テンソル
        """
        x = self.input(x)
        if train:
            x = self.dropout(x)
        x = self.act(x)
        for hidden in self.hiddens:
            x = hidden(x)
            if train:
                x = self.dropout(x)
            x = self.act(x)
        x = self.output(x)
        # x = F.sigmoid(x)
        return x

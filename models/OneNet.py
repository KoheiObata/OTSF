import copy

import torch
from torch import nn

from models import normalization
from models.FSNet import Model as FSNet # Wrap ts2vec-based model with FSNet
from layers.RevIN import RevIN

class OneNet(nn.Module):
    """
    OneNet main class
    - Has backbone (Model_Ensemble, etc.) and synthesizes two series outputs with weights
    - Weights can be dynamically determined by decision function MLP
    - self.weight: Weight parameters for each variable
    - forward: Synthesize and return y1, y2 (two series outputs from backbone)
    """
    def __init__(self, backbone, args):
        super().__init__()
        self.backbone = backbone # Backbone model (Model_Ensemble, etc.)
        # Decision function MLP (for weight estimation)
        self.decision = MLP(n_inputs=args.pred_len * 3, n_outputs=1, mlp_width=32, mlp_depth=3, mlp_dropout=0.1, act=nn.Tanh())
        self.weight = nn.Parameter(torch.zeros(args.enc_in))

    def forward(self, *inp):
        """
        inp: (x, x_mark, loss1, loss2) etc.
        - y1, y2: Two series outputs from backbone
        - inp[-2], inp[-1]: Weights (loss1, loss2)
        Return value: (weighted composite output, y1, y2)
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
    Ensemble model for OneNet
    - encoder: (FSNet_RevIN)
    - encoder_time: backbone (pretrained model like PatchTST)
    - forward_individual: individual output of 2 series
    - forward: final output with weighted sum
    """
    def __init__(self, backbone, args):
        super().__init__()
        _args = copy.deepcopy(args)
        _args.seq_len = 60
        self.seq_len = 60
        self.encoder = normalization.ForecastModel(FSNet(_args), num_features=args.enc_in, seq_len=60)

        args.fsnet_path = f'./results/pretrain/FSNet/RevIN/{args.data}/60_{args.pred_len}/lr0.001/{args.ii}/checkpoints/checkpoint.pth'
        import os
        if os.path.exists(args.fsnet_path):
            print('Load FSNet from', args.fsnet_path)
            checkpoint = torch.load(args.fsnet_path, map_location='cpu')
            self.encoder.load_state_dict(checkpoint['model'])
        else:
            print('FSNet not found', args.fsnet_path)
        self.encoder_time = backbone

    def forward_individual(self, x, x_mark):
        """
        Individual output of 2 series
        - y1: output from encoder_time
        - y2: output from encoder(FSNet)
        """
        y1 = self.encoder_time(x, x_mark)
        y2 = self.encoder.forward(x[..., -self.seq_len:, :], x_mark[..., -self.seq_len:, :] if x_mark is not None else None)
        return y1, y2

    def forward(self, x, x_mark, w1=0.5, w2=0.5):
        """
        Synthesize output of 2 series with weighted sum
        - w1, w2: weights for each series
        Return value: (final output, y1, y2)
        """
        y1, y2 = self.forward_individual(x, x_mark)
        return y1 * w1 + y2 * w2, y1, y2

    def store_grad(self):
        """
        Store gradients of PadConv layers of two encoders
        """
        for name, layer in self.encoder.named_modules():
            if 'PadConv' in type(layer).__name__:
                layer.store_grad()
        for name, layer in self.encoder_time.named_modules():
            if 'PadConv' in type(layer).__name__:
                layer.store_grad()

    def try_trigger_(self, flag=True):
        """
        Switch trigger flag of PadConv layers of two encoders
        """
        for name, layer in self.encoder.named_modules():
            if 'PadConv' in type(layer).__name__:
                layer.try_trigger = flag
        for name, layer in self.encoder_time.named_modules():
            if 'PadConv' in type(layer).__name__:
                layer.try_trigger = flag

class MLP(nn.Module):
    """
    Generic multi-layer perceptron (for decision function)
    - n_inputs: input dimension
    - n_outputs: output dimension
    - mlp_width: hidden layer width
    - mlp_depth: number of layers
    - mlp_dropout: dropout rate
    - act: activation function
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
        x: input tensor
        train: dropout activation flag
        Return value: output tensor
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

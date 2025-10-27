import torch
import torch.nn as nn

from layers.RevIN import RevIN
from layers.ts2vec.encoder import TS2VecEncoderWrapper
from layers.ts2vec.fsnet import TSEncoder

class Model(nn.Module):
    """
    FSNet (Fast and Slow Network) model body
    - Time series feature extraction using TS2Vec-based encoder
    - Input: x (B, T, D), x_mark (B, T, time_features)
    - Output: prediction values (B, pred_len, c_out)
    - store_grad/try_trigger_ are special features of PadConv layer
    """
    def __init__(self, args):
        super().__init__()
        # Build TS2Vec encoder
        encoder = TSEncoder(input_dims=args.enc_in + (4 if args.timeenc == 1 else 7),
                            output_dims=320,  # Standard value for ts2vec
                            hidden_dims=64,   # Standard value for ts2vec
                            depth=10)
        self.encoder = TS2VecEncoderWrapper(encoder, mask='all_true')
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.dim = args.c_out * args.pred_len
        # Fully connected layer for prediction
        self.regressor = nn.Linear(320, self.dim)

    def forward(self, x, x_mark=None):
        """
        x: Input sequence (B, T, D)
        x_mark: Time features (B, T, time_features)
        Return value: prediction values (B, pred_len, c_out)
        """
        if x_mark is None:
            x_mark = torch.zeros(*x.shape[:2], 7, device=x.device)
        x = torch.cat([x, x_mark], dim=-1)
        rep = self.encoder(x)  # Feature extraction
        y = self.regressor(rep)
        y = y.reshape(len(y), self.pred_len, -1)
        return y

    def store_grad(self):
        """
        Save gradients in PadConv layer (special feature of FSNet)
        """
        for name, layer in self.encoder.named_modules():
            if 'PadConv' in type(layer).__name__:
                layer.store_grad()

    def try_trigger_(self, flag=True):
        """
        Toggle trigger flag in PadConv layer (special feature of FSNet)
        """
        for name, layer in self.encoder.named_modules():
            if 'PadConv' in type(layer).__name__:
                layer.try_trigger = flag


class Model_Ensemble(Model):
    """
    Ensemble extension model of FSNet
    - Has two encoders: input sequence direction and time direction
    - RevIN normalization (optional)
    - forward_individual: Individual outputs of two sequences
    - forward: Final output with weighted sum
    """
    def __init__(self, args):
        super().__init__(args)
        self.norm = False
        if args.normalization.lower() == 'revin':
            self.norm = True
            self.revin = RevIN(num_features=args.enc_in)
        depth = 10
        # Encoder in time direction
        encoder = TSEncoder(input_dims=args.seq_len,
                            output_dims=320,  # Standard value for ts2vec
                            hidden_dims=64,   # Standard value for ts2vec
                            depth=depth)
        self.encoder_time = TS2VecEncoderWrapper(encoder, mask='all_true')
        self.regressor_time = nn.Linear(320, args.pred_len)

    def forward_individual(self, x, x_mark=None):
        """
        Individual prediction in two series directions
        - y1: output from time direction encoder (B, c_out, pred_len)
        - y2: normal FSNet output (B, pred_len, c_out)
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
        Synthesize output of two series directions with weighted sum
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
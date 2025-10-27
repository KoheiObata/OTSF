import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    """
    three-layered MLP with a hidden dimension of 16 as the student model,
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len + configs.pred_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.hidden_dim = 16

        # Input dimension: seq_len + pred_len (concatenate teacher output and original input data)
        input_dim = self.seq_len * self.channels
        output_dim = self.pred_len * self.channels

        # Build 3-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),  # Layer 1: input → hidden
            nn.ReLU(),                              # Activation function
            nn.Linear(self.hidden_dim, self.hidden_dim),  # Layer 2: hidden → hidden
            nn.ReLU(),                              # Activation function
            nn.Linear(self.hidden_dim, output_dim)  # Layer 3: hidden → output
        )

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        batch_size, seq_len, channels = x.shape

        # Flatten input data: [Batch, Input length, Channel] -> [Batch, Input length * Channel]
        x_reshaped = x.reshape(batch_size, seq_len * channels)  # [Batch, Input length * Channel]
        output_reshaped = self.mlp(x_reshaped)  # [Batch, Output length * Channel]
        output = output_reshaped.reshape(batch_size, self.pred_len, channels)  # [Batch, Output length, Channel]

        return output  # [Batch, Output length, Channel]
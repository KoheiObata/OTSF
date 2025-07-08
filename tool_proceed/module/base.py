# base.py: 適応層の基底クラス
import torch.nn as nn


class Adaptation(object):
    """
    適応層の基底クラス。
    - flag_adapt_bias: バイアス適応を有効化するか
    - flag_adapt_weight: 重み適応を有効化するか
    - merge_weights: 適応重みをマージするか
    - freeze_weight, freeze_bias: 重み・バイアスの学習可否
    - assign_adaptation: サブクラスで実装必須。適応パラメータを割り当てる。
    """
    def __init__(self, flag_adapt_bias: bool, flag_adapt_weight: bool = True,
                 merge_weights: bool = True, freeze_weight: bool = True, freeze_bias: bool = True):
        assert isinstance(self, nn.Module)
        self.flag_adapt_bias = flag_adapt_bias
        self.flag_adapt_weight = flag_adapt_weight
        self.weight.requires_grad = not freeze_weight
        if self.bias is not None:
            self.bias.requires_grad = not freeze_bias
        # マージ済みフラグ
        self.merged = False
        self.merge_weights = merge_weights

    def assign_adaptation(self, adaptation):
        """
        適応パラメータを割り当てる（サブクラスで実装）
        """
        raise NotImplementedError

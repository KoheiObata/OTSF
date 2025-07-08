# proceed.py: オンライン適応学習用ラッパーモデル
import torch
import torch.nn as nn
import transformers
from tool_proceed.module.generator import AdaptGenerator
from tool_proceed.module import down_up


def normalize(W, max_norm=1):
    """
    テンソルWのノルムをmax_norm以下に正規化
    - W: 任意のshapeのテンソル
    - max_norm: 許容する最大ノルム
    返り値: ノルム制限後のテンソル
    """
    W_norm = torch.norm(W, dim=-1, keepdim=True)  # 最終次元でノルム計算
    scale = torch.clip(max_norm / W_norm, max=1)  # ノルムがmax_normを超えた場合のみスケール
    return W * scale


class Transpose(nn.Module):
    """
    テンソルの次元を入れ替えるラッパー
    - dims: 入れ替える次元のタプル
    - contiguous: メモリ連続化するか
    """
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        # 指定次元を入れ替え、必要ならcontiguousでメモリ連続化
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class Proceed(nn.Module):
    """
    オンライン適応学習のためのラッパーモデル
    - backbone: 元の時系列モデル
    - generator: 各層の適応パラメータ生成器
    - recent_batch: 直近系列のバッファ（オンライン適応用）
    - mlp1, mlp2: 系列→コンセプト特徴量変換MLP
    - flag_xxx: オンライン学習や更新の制御フラグ
    """
    def __init__(self, backbone, args):
        super().__init__()
        self.args = args
        if args.freeze:
            backbone.requires_grad_(False)  # モデル全体の重みを固定
        self.backbone = add_adapters_(backbone, args)  # 指定層にadapterを挿入
        self.more_bias = not args.freeze  # freeze時はバイアス学習しない
        # 各層の適応パラメータ生成器（MLP）
        self.generator = AdaptGenerator(backbone, args.concept_dim,
                                        activation=nn.Sigmoid if args.act == 'sigmoid' else nn.Identity,
                                        adaptive_dim=False, need_bias=self.more_bias,
                                        shared=not args.individual_generator,
                                        mid_dim=args.bottleneck_dim)
        # recent_batch: 直近系列（バッチサイズ1, 入力長+予測長, 入力次元）
        self.register_buffer('recent_batch', torch.zeros(1, args.seq_len + args.pred_len, args.enc_in), persistent=False)
        if args.ema > 0:
            # recent_concept: 直近系列のコンセプト特徴量（EMAで滑らかに更新）
            self.register_buffer('recent_concept', None, persistent=True)
        # mlp1: 入力系列（[B, L, C]）→コンセプト特徴量（[B, C, D]）
        self.mlp1 = nn.Sequential(
            Transpose(-1, -2),  # [B, L, C]→[B, C, L]
            nn.Linear(args.seq_len, args.concept_dim),  # L→D
            nn.GELU(),
            nn.Linear(args.concept_dim, args.concept_dim)  # D→D
        )
        # mlp2: recent_batch用（[B, L+P, C]→[B, C, D]）
        self.mlp2 = nn.Sequential(
            Transpose(-1, -2),
            nn.Linear(args.seq_len + args.pred_len, args.concept_dim),
            nn.GELU(),
            nn.Linear(args.concept_dim, args.concept_dim)
        )
        self.ema = args.ema  # EMA係数
        # オンライン学習・更新・推論制御用フラグ
        self.flag_online_learning = False
        self.flag_update = False
        self.flag_current = False
        self.flag_basic = False

    def generate_adaptation(self, x):
        """
        現在系列xとrecent_batchからコンセプト特徴量を抽出し、
        差分（drift）をgeneratorに入力して適応パラメータを生成
        - x: 入力系列（[B, L, C]）
        - recent_batch: 直近系列（[B, L+P, C]）
        - concept: 現在系列の特徴量（[B, C, D]→平均で[B, D]）
        - recent_concept: 直近系列の特徴量（[B, C, D]→平均で[D]）
        - drift: concept - recent_concept（[B, D]）
        - generator: driftから各層の適応パラメータを生成
        - EMA: recent_conceptを滑らかに更新（flag_updateやflag_online_learningで制御）
        """
        concept = self.mlp1(x).mean(-2)  # 入力系列から特徴量抽出→系列方向平均（[B, D]）
        # recent_batchから特徴量抽出→系列方向・バッチ方向平均（[D]）
        recent_concept = self.mlp2(self.recent_batch).mean(-2).mean(list(range(0, self.recent_batch.dim() - 2)))
        if self.ema > 0:
            if self.recent_concept is not None:
                # EMAでrecent_conceptを滑らかに更新
                recent_concept = self.recent_concept * self.ema + recent_concept * (1 - self.ema)
            # flag_updateまたはオンライン学習時にrecent_conceptを更新
            if self.flag_update or self.flag_online_learning and not self.flag_current:
                self.recent_concept = recent_concept.detach()
        drift = concept - recent_concept  # 現在と直近の差分
        res = self.generator(drift, need_clip=not self.args.wo_clip)  # driftから各層の適応パラメータ生成
        return res

    def forward(self, *x):
        """
        各層に適応パラメータを割り当ててbackboneを実行
        - flag_basic: 適応なしでバイアスのみ返す（ベースライン用）
        - adaptations: 各層の適応パラメータ（dict: 層名→パラメータ）
        - assign_adaptation: 各層のadapterにパラメータをセット
        """
        if self.flag_basic:
            # ベースライン（適応なし）: 各層のバイアスのみ返す
            adaptations = {}
            for i, (k, adapter) in enumerate(self.generator.bottlenecks.items()):
                # adapter.need_biasがTrueならバイアス、FalseならNone
                adaptations[k] = adapter.biases[-1] if adapter.need_bias else [None] * len(self.generator.dim_name_dict[k])
        else:
            # 通常: generate_adaptationで各層のパラメータ生成
            adaptations = self.generate_adaptation(x[0])
        # 各層ごとにassign_adaptationでパラメータをセット
        for out_dim, adaptation in adaptations.items():
            for i in range(len(adaptation)):
                name = self.generator.dim_name_dict[out_dim][i]  # 層名（backbone内のパス）
                self.backbone.get_submodule(name).assign_adaptation(adaptation[i])

        return self.backbone(*x)

    def freeze_adapter(self, freeze=True):
        """
        適応層の学習可否を切り替え
        - freeze=True: adapterの重み・バイアスを固定
        - freeze=False: adapterの重み・バイアスを学習可
        - mlp1, mlp2, 各adapterのweights/biasesに対してrequires_gradとzero_gradを設定
        """
        for module_name in ['mlp1', 'mlp2']:
            if hasattr(self, module_name):
                getattr(self, module_name).requires_grad_(not freeze)
                getattr(self, module_name).zero_grad(set_to_none=True)
        for adapter in self.generator.bottlenecks.values():
            adapter.weights.requires_grad_(not freeze)
            adapter.weights.zero_grad(set_to_none=True)
            # adapter.biasesのうちweightsと同数分（通常はバイアス項）
            adapter.biases[:len(adapter.weights) - 1].requires_grad_(not freeze)
            adapter.biases[:len(adapter.weights) - 1].zero_grad(set_to_none=True)

    def freeze_bias(self, freeze=True):
        """
        バイアス項の学習可否を切り替え
        - more_bias=Trueのときのみ有効
        - 各adapterのバイアス項（biases[-1]）に対してrequires_gradとzero_gradを設定
        """
        if self.more_bias:
            for adapter in self.generator.bottlenecks.values():
                adapter.biases[-1].requires_grad_(not freeze)
                adapter.biases[-1:].zero_grad(set_to_none=True)


def add_adapters_(parent_module: nn.Module, args, top_level=True):
    """
    モデル全体にadapterを再帰的に挿入するユーティリティ
    - parent_module: 対象となる親モジュール
    - args.tune_mode: 挿入対象層の選択（all_down_up, down_up, それ以外は再帰）
    - top_level: 再帰呼び出しかどうか
    - down_up.add_down_up_: 指定層をDown_Up適応層に置換
    - 再帰的に全子モジュールを探索
    """
    for name, module in parent_module.named_children():
        if args.tune_mode == 'all_down_up' and isinstance(module, (nn.Conv1d, nn.Linear, transformers.Conv1D,
                                                                     nn.LayerNorm, nn.BatchNorm1d)):
            down_up.add_down_up_(parent_module, name, freeze_weight=args.freeze, merge_weights=args.merge_weights,)
        elif args.tune_mode == 'down_up' and isinstance(module, (nn.Conv1d, nn.Linear, transformers.Conv1D)):
            down_up.add_down_up_(parent_module, name, freeze_weight=args.freeze, merge_weights=args.merge_weights,)
        else:
            # 子モジュールに対して再帰的にadapter挿入
            add_adapters_(module, args, False)
    return parent_module

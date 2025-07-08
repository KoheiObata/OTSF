import torch
from torch import optim, nn
import torch.nn.functional as F

from exp import Exp_Online
from models.OneNet import OneNet, Model_Ensemble


class Exp_FSNet(Exp_Online):
    """
    FSNet（Fast and Slow Network）用のオンライン学習クラス
    - FSNetの特殊な機能（勾配保存、トリガー機能）に対応
    """
    def __init__(self, args):
        super().__init__(args)

    def _update(self, *args, **kwargs):
        """
        更新時に勾配を保存（FSNetの機能）
        """
        ret = super()._update(*args, **kwargs)
        if hasattr(self.model, 'store_grad'):
            self.model.store_grad()
        return ret

    def vali(self, *args, **kwargs):
        """
        バリデーション時にトリガー機能を有効化
        """
        if not hasattr(self.model, 'try_trigger_'):
            return super().vali(*args, **kwargs)
        else:
            self.model.try_trigger_(True)
            ret = super().vali(*args, **kwargs)
            self.model.try_trigger_(False)
            return ret

    def online(self, *args, **kwargs):
        """
        オンライン学習時にトリガー機能を有効化
        """
        if not hasattr(self.model, 'try_trigger_'):
            return super().online(*args, **kwargs)
        else:
            self.model.try_trigger_(True)
            ret = super().online(*args, **kwargs)
            self.model.try_trigger_(False)
            return ret

    def analysis_online(self):
        """
        分析時にトリガー機能を有効化
        """
        if hasattr(self.model, 'try_trigger_'):
            self.model.try_trigger_(True)
        return super().analysis_online()


class Exp_OneNet(Exp_FSNet):
    """
    OneNet用のオンライン学習クラス
    - OneNetの特殊な構造（重みとバイアスの分離最適化）に対応
    - アンサンブル学習機能も含む
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 重みとバイアス用の別々のオプティマイザー
        self.opt_w = optim.Adam([self.model.weight], lr=self.args.learning_rate_w)
        self.opt_bias = optim.Adam(self.model.decision.parameters(), lr=self.args.learning_rate_bias)
        self.bias = torch.zeros(self.args.enc_in, device=self.model.weight.device)

    def _select_optimizer(self, filter_frozen=True, return_self=True, model=None):
        """
        オプティマイザー選択時にOneNetの特殊構造に対応
        """
        if model is None or isinstance(model, OneNet):
            return super()._select_optimizer(filter_frozen, return_self, model=self.model.backbone)
        return super()._select_optimizer(filter_frozen, return_self, model=model)

    def state_dict(self, *args, **kwargs):
        """
        状態保存時に重みとバイアスのオプティマイザーも含める
        """
        destination = super().state_dict(*args, **kwargs)
        destination['opt_w'] = self.opt_w.state_dict()
        destination['opt_bias'] = self.opt_bias.state_dict()
        return destination

    # def load_state_dict(self, state_dict, model=None):
    #     self.model.bias.data = state_dict['model']['bias']
    #     return super().load_state_dict(state_dict, model)

    def _build_model(self, model=None, framework_class=None):
        """
        モデル構築時にアンサンブル機能を追加
        """
        if self.args.model not in ['TCN', 'FSNet', 'TCN_Ensemble', 'FSNet_Ensemble']:
            framework_class = [Model_Ensemble, OneNet]
        else:
            framework_class = OneNet
        return super()._build_model(model, framework_class=framework_class)

    def train_loss(self, criterion, batch, outputs):
        """
        OneNetの損失計算（複数出力の組み合わせ）
        """
        return super().train_loss(criterion, batch, outputs[1]) + super().train_loss(criterion, batch, outputs[2])

    def vali(self, vali_data, vali_loader, criterion):
        """
        バリデーション時にバイアスをリセット
        """
        self.bias = torch.zeros(self.args.enc_in, device=self.model.weight.device)
        ret = super().vali(vali_data, vali_loader, criterion)
        self.phase = None
        return ret

    def update_valid(self, valid_data=None):
        """
        バリデーション更新時にバイアスをリセット
        """
        self.bias = torch.zeros(self.args.enc_in, device=self.model.weight.device)
        return super().update_valid(valid_data)

    def forward(self, batch):
        """
        OneNetの順伝播処理
        - オンライン学習時は動的な重みとバイアスを使用
        - 通常時は固定の重みを使用
        """
        b, t, d = batch[1].shape
        if hasattr(self, 'phase') and self.phase in self.online_phases:
            # オンライン学習時：動的な重みとバイアス
            weight = self.model.weight.view(1, 1, -1).repeat(b, t, 1)
            bias = self.bias.view(-1, 1, d)
            loss1 = F.sigmoid(weight + bias.repeat(1, t, 1)).view(b, t, d)
        else:
            # 通常時：固定の重み
            loss1 = F.sigmoid(self.model.weight).view(1, 1, -1)
            loss1 = loss1.repeat(b, t, 1)
        batch = batch + [loss1, 1 - loss1]  # 重みをバッチに追加
        return super().forward(batch)

    def _update(self, batch, criterion, optimizer, scaler=None):
        """
        OneNetの更新処理（重みとバイアスの分離最適化）
        """
        batch_y = batch[1]
        b, t, d = batch_y.shape

        # 通常の更新
        loss, (outputs, y1, y2) = super()._update(batch, criterion, optimizer, scaler)

        # 重みの更新
        loss_w = criterion(outputs, batch_y)
        loss_w.backward()
        self.opt_w.step()
        self.opt_w.zero_grad()

        # バイアスの更新
        y1_w, y2_w = y1.detach(), y2.detach()
        true_w = batch_y.detach()
        loss1 = F.sigmoid(self.model.weight).view(1, 1, -1)
        loss1 = loss1.repeat(b, t, 1)
        inputs_decision = torch.cat([loss1 * y1_w, (1 - loss1) * y2_w, true_w], dim=1)
        bias = self.model.decision(inputs_decision.permute(0, 2, 1)).view(b, 1, -1)
        weight = self.model.weight.view(1, 1, -1).repeat(b, t, 1)
        loss1 = F.sigmoid(weight + bias.repeat(1, t, 1))
        loss2 = 1 - loss1
        loss_bias = criterion(loss1 * y1_w + loss2 * y2_w, true_w)
        loss_bias.backward()
        self.opt_bias.step()
        self.opt_bias.zero_grad()

        return loss / 2, outputs

    def _update_online(self, batch, criterion, optimizer, scaler=None):
        """
        OneNetのオンライン更新処理
        - 重みとバイアスを順次更新
        """
        batch_y = batch[1]
        b, t, d = batch_y.shape

        # 通常の更新
        loss, (outputs, y1, y2) = super()._update(batch, criterion, optimizer, scaler)

        # バイアスの更新
        y1_w, y2_w = y1.detach(), y2.detach()
        loss1 = F.sigmoid(self.model.weight).view(1, 1, -1).repeat(b, t, 1)
        inputs_decision = torch.cat([loss1 * y1_w, (1 - loss1) * y2_w, batch_y], dim=1)
        self.bias = self.model.decision(inputs_decision.permute(0, 2, 1))
        weight = self.model.weight.view(1, 1, -1).repeat(b, t, 1)
        bias = self.bias.view(b, 1, -1)
        loss1 = F.sigmoid(weight + bias.repeat(1, t, 1))
        loss2 = 1 - loss1

        outputs_bias = loss1 * y1_w + loss2 * y2_w
        loss_bias = criterion(outputs_bias, batch_y)
        loss_bias.backward()
        self.opt_bias.step()
        self.opt_bias.zero_grad()

        # 重みの更新
        loss1 = F.sigmoid(self.model.weight).view(1, 1, -1)
        loss1 = loss1.repeat(b, t, 1)
        loss_w = criterion(loss1 * y1_w + (1 - loss1) * y2_w, batch_y)
        loss_w.backward()
        self.opt_w.step()
        self.opt_w.zero_grad()
        return loss / 2, outputs

import torch
from exp import Exp_Online
from util.buffer import Buffer



class Exp_ER(Exp_Online):
    """
    Experience Replay（経験再生）を実装したオンライン学習クラス
    - 過去のデータをバッファに保存し、忘れることを防ぐ
    - バッファサイズ500で過去データを管理
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.count = 0
        self.alpha = 0.2

    def train_loss(self, criterion, batch, outputs):
        """
        経験再生を含む損失計算
        - 通常の損失に加えて、バッファからの過去データでの損失も追加
        """
        loss = super().train_loss(criterion, batch, outputs)
        if not self.buffer.is_empty():
            buff = self.buffer.get_data(self.args.mini_batch)  # バッファからmini_batch個のサンプル取得
            # buff[0]は入力データ、buff[1]はラベルデータ, buff[2]は入力の時間特徴量，buff[3]は出力の時間特徴量
            out = self.forward(buff[:-1])   # buff[:-1]は「最後の属性（インデックス）を除いたデータ」
            if isinstance(outputs, (tuple, list)):
                out = out[0]
            loss += self.alpha * criterion(out, buff[1])  # バッファデータでの損失を0.2倍で追加
        return loss

    def _update_online(self, batch, criterion, optimizer, scaler=None):
        """
        オンライン更新時にバッファにデータを追加
        """
        loss, outputs = self._update(batch, criterion, optimizer, scaler)
        idx = self.count + torch.arange(batch[1].size(0)).to(self.device)
        self.count += batch[1].size(0)
        self.buffer.add_data(*batch, idx)  # バッファにデータとインデックスを追加
        return loss, outputs


class Exp_DERpp(Exp_Online):
    """
    Dark Experience Replay++（DER++）を実装したオンライン学習クラス
    - ERを拡張し、より効果的な経験再生を実現
    - 予測出力もバッファに保存して活用
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.count = 0
        self.alpha = 0.2
        self.beta = 0.2

    def train_loss(self, criterion, batch, outputs):
        """
        DER++の損失計算
        - 通常の損失に加えて、バッファからの予測出力での損失も追加
        """
        loss = super().train_loss(criterion, batch, outputs)
        if not self.buffer.is_empty():
            buff = self.buffer.get_data(self.args.mini_batch)
            out = self.forward(buff[:-1])
            if isinstance(outputs, (tuple, list)):
                out = out[0]
            loss += self.alpha * criterion(buff[1], out)  # 予測出力と，ラベルデータ，の損失を追加
            loss += self.beta * criterion(buff[-1], out)  # 今のモデルの予測出力と，過去のモデルの予測出力，の損失を追加
        return loss

    def _update_online(self, batch, criterion, optimizer, scaler=None):
        """
        オンライン更新時に予測出力もバッファに追加
        """
        loss, outputs = self._update(batch, criterion, optimizer, scaler)
        self.count += batch[1].size(0)
        if isinstance(outputs, (tuple, list)):
            self.buffer.add_data(*(batch + [outputs[0]]))  # 予測出力もバッファに追加
        else:
            self.buffer.add_data(*(batch + [outputs]))
        return loss, outputs

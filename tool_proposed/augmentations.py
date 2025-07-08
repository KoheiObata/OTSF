import numpy as np
import torch


def mixing_coefficient_set_for_each(similarities, args):
    """
    類似度に基づいて混合係数を設定する関数
    - 類似度が閾値（0.8）を超える場合は0.7～1.0の範囲でランダムな係数
    - それ以外は切断正規分布で係数を生成

    Args:
        similarities: torch.Tensor, shape [topk]
        args: 設定パラメータ（mean, std, low_limit, high_limit）

    Returns:
        mixing_coeff: torch.Tensor, shape [topk]
    """
    threshold = 0.8
    similarities = similarities.detach().cpu()
    mixing_coeff = torch.empty_like(similarities)

    # 閾値超えは0.7～1.0の一様乱数
    high_mask = similarities > threshold
    mixing_coeff[high_mask] = 0.7 + 0.3 * torch.rand(high_mask.sum())

    # 閾値以下は切断正規分布
    low_mask = ~high_mask
    if low_mask.sum() > 0:
        tmp = torch.empty(low_mask.sum())
        torch.nn.init.trunc_normal_(tmp, mean=args.mean, std=args.std, a=args.low_limit, b=args.high_limit)
        mixing_coeff[low_mask] = tmp

    return mixing_coeff

def phase_mix(seq_x, buff_x, coeff):
    """
    位相の混合を行う関数
    - 類似度に基づいて位相の混合係数を決定
    - 位相の差分を計算して適切に混合
    - 複数のサンプルを生成（mini_batch分）

    Args:
        seq_x: torch.Size([1, window, features])
        buff_x: torch.Size([topk, window, features])
        coeff: torch.Size([topk])

    Returns:
        mixed_phase: torch.Size([topk, window, features])
    """
    # seq_xをtopk個に拡張
    seq_x_expanded = seq_x.repeat(buff_x.shape[0], 1, 1)

    # 位相の差分を計算
    phase_difference = seq_x_expanded - buff_x
    dtheta = phase_difference % (2 * torch.pi)

    # 位相の差分を-πからπの範囲に正規化
    dtheta[dtheta > torch.pi] -= 2 * torch.pi
    clockwise = dtheta > 0
    sign = torch.where(clockwise, -1, 1)

    # coeffを[batch, 1, 1]にreshape
    coeff = coeff.view(-1, 1, 1).to(seq_x.device)
    # 位相を混合 (coeff*seq_x_expandedにすべきだけど，そうすると精度が低い）
    # mixed_phase = coeff * seq_x_expanded + (1 - coeff) * torch.abs(dtheta) * sign
    mixed_phase = seq_x_expanded + (1 - coeff) * torch.abs(dtheta) * sign
    return mixed_phase

def abs_mix(seq_x, buff_x, coeff):
    """
    絶対値スペクトルのMixup
    Args:
        seq_x: torch.Size([1, window, features])
        buff_x: torch.Size([topk, window, features])
        coeff: torch.Size([topk])
    Returns:
        mixed_abs: torch.Size([topk, window, features])
    """
    # seq_xをtopk個に拡張
    seq_x_expanded = seq_x.repeat(buff_x.shape[0], 1, 1)
    # coeffを[batch, 1, 1]にreshape
    coeff = coeff.view(-1, 1, 1).to(seq_x.device)
    mixed_abs = coeff * seq_x_expanded + (1 - coeff) * buff_x
    return mixed_abs

def fft_mix(seq_x, buff_x, similarities, args):
    """
    類似度ベースの高度なフーリエ領域Mixup手法
    - 類似度に基づいて混合係数を決定
    - 振幅と位相の両方を混合
    - 複数のサンプルを生成（mini_batch分）

    Args:
        seq_x: torch.Size([1, window, features])
        buff_x: torch.Size([topk, window, features])
        similarities: torch.Size([topk])
        args: 設定パラメータ（mini_batch等を含む）

    Returns:
        mixed_samples_time: 混合された時系列データ（複数サンプル）
    """
    device = seq_x.device

    abs_mixing_coeff = mixing_coefficient_set_for_each(similarities, args)
    phase_mixing_coeff = mixing_coefficient_set_for_each(similarities, args)

    # abs_mixing_coeff = 1 - abs_mixing_coeff
    # phase_mixing_coeff = 1 - phase_mixing_coeff
    # abs_mixing_coeff = abs_mixing_coeff/2
    # phase_mixing_coeff = phase_mixing_coeff/2

    seq_x_fft = torch.fft.rfft(seq_x, dim=1, norm='ortho')
    seq_x_abs_fft = torch.abs(seq_x_fft).to(device)
    seq_x_phase_fft = torch.angle(seq_x_fft).to(device)

    buff_x_fft = torch.fft.rfft(buff_x, dim=1, norm='ortho')
    buff_x_abs_fft = torch.abs(buff_x_fft).to(device)
    buff_x_phase_fft = torch.angle(buff_x_fft).to(device)

    mixed_abs = abs_mix(seq_x_abs_fft, buff_x_abs_fft, abs_mixing_coeff)
    mixed_phase = phase_mix(seq_x_phase_fft, buff_x_phase_fft, phase_mixing_coeff)

    z =  torch.polar(mixed_abs, mixed_phase)
    mixed_samples_time = torch.fft.irfft(z, dim=1, norm='ortho')
    return mixed_samples_time


import torch
import torch.nn.functional as F
from scipy.signal import correlate

def gen_new_aug(seq_x, buff_x, base_ratio_start=0.8, base_ratio_end=0.8, base_seq='x', period=48, lags=None, args=None):
    # base_seqがxの場合はbuff_xを基準にしてseq_xをシフトさせる
    # base_seqがyの場合はseq_xを基準にしてbuff_xをシフトさせる
    # base_seqの値は全て使用される，もう一方のseqのずれた分の値はfill_value(0)になる

    if args.detrend:
        moving_avg_seq_x = calculate_moving_average_torch(seq_x, moving_avg_window=period)
        moving_avg_buff_x = calculate_moving_average_torch(buff_x, moving_avg_window=period)
    else:
        moving_avg_seq_x = torch.zeros_like(seq_x)
        moving_avg_buff_x = torch.zeros_like(buff_x)

    detrend_seq_x = seq_x - moving_avg_seq_x
    detrend_buff_x = buff_x - moving_avg_buff_x

    detrend_seq_x_np = detrend_seq_x.clone().cpu().numpy()
    detrend_buff_x_np = detrend_buff_x.clone().cpu().numpy()

    if lags is None:
        if period is None:
            max_lag = 0
            min_lag = -seq_x.shape[1]
        else:
            max_lag = -int(period/2)
            min_lag = max_lag-period
        lags = torch.tensor(calculate_lag_using_ccf_numpy(detrend_seq_x_np, detrend_buff_x_np, max_lag=max_lag, min_lag=min_lag)).to(seq_x.device)
    else:
        lags = lags.to(seq_x.device)

    detrend_summed_output = slide_x_and_sum_with_y_torch(detrend_seq_x, detrend_buff_x, lags, base_ratio_start=base_ratio_start, base_ratio_end=base_ratio_end, base_seq=base_seq)
    if base_seq.lower() == 'x':
        summed_output = detrend_summed_output + moving_avg_buff_x
    elif base_seq.lower() == 'y':
        summed_output = detrend_summed_output + moving_avg_seq_x
    return summed_output


def slide_x_and_sum_with_y_torch( X: torch.Tensor, Y: torch.Tensor, lags: torch.Tensor, fill_value: float = 0.0, base_seq: str = 'X',
                                 base_ratio_start: float = None, base_ratio_end: float = None) -> torch.Tensor:
    """
    Xを各バッチのラグの分だけ時間軸上でスライドさせ、Yとの和を計算します。
    ベクトル化された実装で、for文を使用しません。

    Args:
        X (torch.Tensor): 基準となる時系列データ。形状は (1, time_steps, num_features)。
                          内部では (time_steps, num_features) として扱います。
        Y (torch.Tensor): 比較対象のバッチ時系列データ。形状は (num_batches, time_steps, num_features)。
        lags (torch.Tensor): 各バッチに適用するラグ値のテンソル。形状は (num_batches,)。
                             正のラグはXが未来にシフトし、元の位置にはfill_valueが来ることを意味します。
                             負のラグはXが過去にシフトし、元の位置にはfill_valueが来ることを意味します。
        fill_value (float): ラグによるシフトで生じる空の要素を埋める値。デフォルトは0.0。

    Returns:
        torch.Tensor: スライドされたXとYの和。形状はYと同じ (num_batches, time_steps, num_features)。
    """
    # 入力形状の検証と整形
    if X.ndim == 3 and X.shape[0] == 1:
        X_processed = X[0] # (1, time, feature) -> (time, feature)
    elif X.ndim == 2:
        X_processed = X # (time, feature)
    else:
        raise ValueError("X の形状は (1, time_steps, num_features) または (time_steps, num_features) である必要があります。")

    if Y.ndim != 3:
        raise ValueError("Y の形状は (num_batches, time_steps, num_features) である必要があります。")

    num_time_steps_X, num_features_X = X_processed.shape
    num_batches, num_time_steps_Y, num_features_Y = Y.shape

    if num_time_steps_X != num_time_steps_Y:
        raise ValueError("XとYの時間ステップ数が異なる場合、この操作は非推奨です。同じである必要があります。")
    if num_features_X != num_features_Y:
        raise ValueError("XとYの特徴量次元数が異なります。")
    if lags.ndim != 1 or lags.shape[0] != num_batches:
        raise ValueError("lags の形状は (num_batches,) である必要があります。")


    # 一方のみが指定された場合の処理
    if base_ratio_start is not None and base_ratio_end is None:
        base_ratio_end = base_ratio_start
    elif base_ratio_start is None and base_ratio_end is not None:
        base_ratio_start = base_ratio_end

    Vectorized = True
    if Vectorized:
        # ベクトル化された実装 (高速)
        if base_seq.lower() == 'x':
            return slide_sum_seq_torch_vectorized(X_processed, Y, lags, base_ratio_start=base_ratio_start, base_ratio_end=base_ratio_end)
        elif base_seq.lower() == 'y':
            return slide_sum_seq_torch_vectorized(Y, X_processed.unsqueeze(0).expand_as(Y), lags, base_ratio_start=base_ratio_start, base_ratio_end=base_ratio_end)
        else:
            raise ValueError("base_seq は 'X' または 'Y' である必要があります。")
    else:
        # 結果を格納するテンソル (Yと同じ形状)
        summed_output = torch.zeros_like(Y, dtype=Y.dtype)
        # 各バッチに対してループ処理
        for i in range(num_batches):
            current_lag = lags[i].item() # テンソルからPythonの整数に変換
            if base_seq.lower() == 'x':
                summed_output[i] = slide_sum_seq_torch(X_processed, Y[i], current_lag, base_ratio_start=base_ratio_start, base_ratio_end=base_ratio_end)
            elif base_seq.lower() == 'y':
                summed_output[i] = slide_sum_seq_torch(Y[i], X_processed, current_lag, base_ratio_start=base_ratio_start, base_ratio_end=base_ratio_end)
            else:
                raise ValueError("base_seq は 'X' または 'Y' である必要があります。")

def slide_sum_seq_torch_vectorized(X: torch.Tensor, Y: torch.Tensor, lags: torch.Tensor, fill_value: float = 0.0,
                                  base_ratio_start: float = None, base_ratio_end: float = None) -> torch.Tensor:
    """
    完全ベクトル化：各バッチごとにXをラグ分だけシフトし、Yと重み付き和を計算

    Args:
        X: 入力テンソル
        Y: 基準テンソル
        lags: ラグ値
        fill_value: シフトで生じる空の要素を埋める値
        base_ratio_start: 線形重みの開始値（指定すると線形重みを使用）
        base_ratio_end: 線形重みの終了値
    """
    num_batches, num_time_steps, num_features = Y.shape

    if X.ndim == 2:
        X_expanded = X.unsqueeze(0).expand(num_batches, -1, -1)
    else:
        X_expanded = X

    time_idx = torch.arange(num_time_steps, device=Y.device).unsqueeze(0).expand(num_batches, -1)
    lags_exp = lags.view(-1, 1)
    src_idx = time_idx - lags_exp

    valid_mask = (src_idx >= 0) & (src_idx < num_time_steps)
    src_idx_clipped = src_idx.clamp(0, num_time_steps-1)
    gather_idx = src_idx_clipped.unsqueeze(-1).expand(-1, -1, num_features)
    X_gathered = torch.gather(X_expanded, 1, gather_idx)

    # 重みの計算（線形重みまたは固定重み）
    if base_ratio_start is not None and base_ratio_end is not None:
        # 線形重み: 時間軸に沿って線形補間
        time_weights = torch.linspace(base_ratio_start, base_ratio_end, num_time_steps, device=Y.device)
        base_ratio_weights = time_weights.unsqueeze(0).unsqueeze(-1).expand(num_batches, -1, num_features)
    else:
        # 固定重み（テンソル形式に変換）
        base_ratio_weights = torch.full((num_batches, num_time_steps, num_features), 1.0, device=Y.device, dtype=Y.dtype)

    # 有効な部分だけ重み付き和、無効な部分はYそのまま
    summed_output = torch.where(
        valid_mask.unsqueeze(-1),
        Y * base_ratio_weights + X_gathered * (1 - base_ratio_weights),
        Y
    )
    return summed_output


def slide_sum_seq_torch( X: torch.Tensor, Y: torch.Tensor, current_lag: int, fill_value: float = 0.0,
                        base_ratio_start: float = None, base_ratio_end: float = None) -> torch.Tensor:
    """
    Yをベースに，Xをラグの分だけシフトさせて，Yとの和を計算します。

    Args:
        X (torch.Tensor): (time_steps, num_features)。
        Y (torch.Tensor): (time_steps, num_features)。
        current_lags (int): ラグ値.
                             正のラグはXが未来にシフトし、元の位置にはfill_valueが来ることを意味します。
                             負のラグはXが過去にシフトし、元の位置にはfill_valueが来ることを意味します。
        fill_value (float): ラグによるシフトで生じる空の要素を埋める値。デフォルトは0.0。
        base_ratio_start (float): 線形重みの開始値（指定すると線形重みを使用）
        base_ratio_end (float): 線形重みの終了値
    """

    num_time_steps, num_features = X.shape
    # Xをラグの分だけシフト
    # torch.rollは循環シフトを行うため、空いた部分をfill_valueで埋める工夫が必要
    X_shifted = torch.full_like(X, fill_value, dtype=X.dtype)
    summed_output = torch.zeros_like(Y, dtype=Y.dtype)

    # 重みの計算（線形重みまたは固定重み）
    if base_ratio_start is not None and base_ratio_end is not None:
        # 線形重み: 時間軸に沿って線形補間
        time_weights = torch.linspace(base_ratio_start, base_ratio_end, num_time_steps, device=Y.device)
        base_ratio_weights = time_weights.unsqueeze(-1).expand(-1, num_features)
    else:
        # 固定重み（テンソル形式に変換）
        base_ratio_weights = torch.full((num_time_steps, num_features), 1.0, device=Y.device, dtype=Y.dtype)

    if current_lag == 0:
        X_shifted = X
        # Yの現在のバッチとシフトされたXを加算
        summed_output = Y * base_ratio_weights + X_shifted * (1-base_ratio_weights)
    elif current_lag > 0: # Xを未来にシフト (右にずらす)
        # 例: lag=2 なら [_, _, x0, x1, x2, ...]
        X_shifted[current_lag:, :] = X[:-current_lag, :]
        summed_output[current_lag:, :] = Y[current_lag:, :] * base_ratio_weights[current_lag:, :] + X_shifted[current_lag:, :] * (1-base_ratio_weights[current_lag:, :])
        summed_output[:current_lag, :] = Y[:current_lag:, :]
    else: # current_lag < 0 (Xを過去にシフト / 左にずらす)
        # 例: lag=-2 なら [x2, x3, ..., _, _]
        X_shifted[:num_time_steps + current_lag, :] = X[-current_lag:, :] # current_lagは負なので -current_lag は正
        end_idx = num_time_steps + current_lag
        summed_output[:end_idx, :] = Y[:end_idx, :] * base_ratio_weights[:end_idx, :] + X_shifted[:end_idx, :] * (1-base_ratio_weights[:end_idx, :])
        summed_output[end_idx:, :] = Y[end_idx:, :]
    return summed_output

def calculate_moving_average_torch(Y: torch.Tensor, moving_avg_window: int) -> torch.Tensor:
    # ... (前略: 入力検証とY_reshapedの作成は同じ) ...

    num_batches, time_steps, num_features = Y.shape # Y.shapeから再取得

    # データを (num_batches * num_features, 1, time_steps) にリシェイプ
    Y_reshaped = Y.permute(0, 2, 1).reshape(-1, 1, time_steps)

    # パディングの計算を修正
    # avg_pool1d は左側に 'padding' の数だけ、右側に 'padding' の数だけパディングを行う
    # 出力長を 'time_steps' にするためには、適切なパディングが必要
    # 例えば、'reflection' パディングは信号の端を反映させるため、より自然な境界処理になる

    # ウィンドウが偶数か奇数かでパディングの数を調整
    # (L_out - L_in) / 2 の部分を計算。L_out は L_in と同じにしたい。
    # output_length = floor((input_length + 2 * padding - kernel_size) / stride + 1)
    # input_length = output_length = time_steps, stride = 1
    # time_steps = floor(time_steps + 2 * padding - kernel_size + 1)
    # これを満たす padding を探す

    # 一般的な「same」パディングは、入力長と同じ出力長を目指します。
    # F.pad と F.avg_pool1d を組み合わせる方法が確実です。
    # 必要なパディングの総数 = kernel_size - 1
    total_padding = moving_avg_window - 1

    # 左側のパディングと右側のパディング
    # 例えばウィンドウ5なら総パディング4。左2、右2。
    # 例えばウィンドウ4なら総パディング3。左1、右2（または左2、右1）。
    # np.convolve(mode='same')は奇数ウィンドウでは中央揃え、偶数ウィンドウでは左に1つずれる
    # PyTorchのavg_pool1dでそれを再現する場合、パディングを非対称にする

    pad_left = total_padding // 2
    pad_right = total_padding - pad_left # 残りを右に

    # F.pad を使って手動でパディングを適用
    # (バッチ, チャンネル, シーケンス長) に対して、(左パディング, 右パディング)
    Y_padded = F.pad(Y_reshaped, (pad_left, pad_right), mode='replicate') # 'replicate'は端の値を繰り返す

    # 移動平均の計算
    # kernel_size=moving_avg_window, stride=1
    # padding は F.pad で行ったため、ここでは0または省略
    Y_moving_avg_reshaped = F.avg_pool1d(
        Y_padded,
        kernel_size=moving_avg_window,
        stride=1,
        padding=0 # F.pad で既に行ったため、ここでは0
    )

    # 出力長が time_steps と必ず一致することを確認
    # もし Y_moving_avg_reshaped.shape[2] != time_steps の場合は
    # 中央を切り取るなどの追加処理が必要になるが、
    # 'replicate'パディングとkernel_sizeの組み合わせで通常は合うはず。

    # 元の形状 (batch, time, feature) に戻す
    Y_moving_avg = Y_moving_avg_reshaped.reshape(num_batches, num_features, time_steps).permute(0, 2, 1)

    return Y_moving_avg


def calculate_lag_using_ccf_numpy(X: np.ndarray, Y: np.ndarray, max_lag: int = None, min_lag: int = None) -> np.ndarray:
    """
    X (1, time, feature) と Y (batch, time, feature) の間で、
    X を基準とした Y の各バッチにおける最適なラグを交差相関関数 (CCF) により計算します。

    Args:
        X (np.ndarray): 基準となる時系列データ。形状は (1, time_steps, num_features)。
                        実際には (time_steps, num_features) として扱います。
        Y (np.ndarray): 比較対象のバッチ時系列データ。形状は (num_batches, time_steps, num_features)。
        max_lag (int, optional): 考慮する最大ラグ。指定しない場合、time_steps // 2 となります。
        min_lag (int, optional): 考慮する最小ラグ。指定しない場合、-time_steps // 2 となります。

    Returns:
        np.ndarray: 各バッチにおける最適なラグの配列 (num_batches, )。
                    返されるラグは「YがXにどれだけ先行するか」を示します。
                    正の値: YがXより未来にずれている（XがYより過去にずれている）
                    負の値: YがXより過去にずれている（XがYより未来にずれている）
    """

    # 入力形状の検証と整形
    if X.ndim == 3 and X.shape[0] == 1:
        X_processed = X[0] # (1, time, feature) -> (time, feature)
    elif X.ndim == 2:
        X_processed = X # (time, feature)
    else:
        raise ValueError("X の形状は (1, time, feature) または (time, feature) である必要があります。")

    if Y.ndim != 3:
        raise ValueError("Y の形状は (batch, time, feature) である必要があります。")

    num_time_steps_X, num_features_X = X_processed.shape
    num_batches, num_time_steps_Y, num_features_Y = Y.shape

    if num_time_steps_X != num_time_steps_Y:
        print("警告: XとYの時間ステップ数が異なります。短い方に合わせて切り詰めます。")
        min_time_steps = min(num_time_steps_X, num_time_steps_Y)
        X_processed = X_processed[:min_time_steps, :]
        Y = Y[:, :min_time_steps, :]
        num_time_steps = min_time_steps
    else:
        num_time_steps = num_time_steps_X

    if num_features_X != num_features_Y:
        raise ValueError("XとYの特徴量次元数が異なります。")

    num_features = num_features_X

    if max_lag is None:
        max_lag = num_time_steps // 2
    if min_lag is None:
        min_lag = -num_time_steps // 2
    if min_lag > max_lag:
        raise ValueError("min_lagはmax_lagより大きい必要があります。")

    optimal_lags_per_batch = np.zeros(num_batches, dtype=int)

    # 各バッチについて処理
    for i in range(num_batches):
        y_batch = Y[i] # 現在のバッチ (time, feature)

        # 全フィーチャーペアのCCFを格納するリスト
        feature_pair_ccfs = []

        # 各特徴量ペアについてCCFを計算
        for f in range(num_features):
            x_feature = X_processed[:, f]
            y_feature = y_batch[:, f]

            # scipy.signal.correlate を使用
            # 'full'モードで計算されるCCFの長さは (len(x) + len(y) - 1)
            # ラグは -(len(y)-1) から (len(x)-1) まで
            correlation = correlate(x_feature, y_feature, mode='full')

            # 相関係数に正規化（オプションだが推奨）
            # ここでは簡単のため、最大値で正規化する代わりに、
            # Pearson相関係数を計算する関数を自作するか、
            # scipy.stats.pearsonr を各ラグで適用する方が厳密
            # ここでは scipy.signal.correlate が返す相関値をそのまま使うが、
            # これは厳密な相関係数ではないことに注意。ピーク位置は見つけられる。

            # ラグ軸の生成
            # correlate('full') のラグは -(len(y)-1) から (len(x)-1)
            # 今回は X と Y の長さが同じなので -(N-1) から (N-1)
            lags = np.arange(-num_time_steps + 1, num_time_steps)

            # max_lag で範囲を制限
            # ラグのインデックスを計算してスライス
            min_idx = np.where(lags == min_lag)[0]
            max_idx = np.where(lags == max_lag)[0]

            if len(min_idx) == 0 or len(max_idx) == 0:
                # max_lagが時系列の長さを超えている場合など、lagsに含まれないケース
                # この場合は full correlation を全て使うか、エラーを出すか
                # 簡単のためここでは全範囲のラグを使用する
                pass
            else:
                correlation = correlation[min_idx[0] : max_idx[0] + 1]
                lags = lags[min_idx[0] : max_idx[0] + 1]

            feature_pair_ccfs.append(correlation)

        if not feature_pair_ccfs:
            optimal_lags_per_batch[i] = np.nan # データがない場合
            continue

        # 各フィーチャーのCCFを合計または平均して、バッチ全体のCCFを生成
        # 例: 合計の相関を使う
        batch_total_ccf = np.sum(feature_pair_ccfs, axis=0)

        # 最も相関が高くなるラグを特定
        # np.argmax は最大値の最初のインデックスを返す
        best_lag_idx = np.argmax(batch_total_ccf)
        optimal_lag = lags[best_lag_idx]

        optimal_lags_per_batch[i] = optimal_lag

    return optimal_lags_per_batch


def test_vectorized_implementation():
    """
    ベクトル化された実装が元の実装と同じ結果を返すかテストする関数
    """
    import torch

    # テストデータの作成
    torch.manual_seed(42)
    num_batches = 4
    num_time_steps = 10
    num_features = 3

    # テストケース1: 単一のXテンソル
    X = torch.randn(num_time_steps, num_features)
    Y = torch.randn(num_batches, num_time_steps, num_features)
    lags = torch.tensor([0, 2, -1, 3])  # 異なるラグ値

    print("Test data:")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"lags: {lags}")
    print(f"X:\n{X}")
    print(f"Y:\n{Y}")

    # 元の実装で各バッチを処理（固定重み）
    original_results = []
    for i in range(num_batches):
        result = slide_sum_seq_torch(X, Y[i], lags[i].item(), base_ratio_start=0.5, base_ratio_end=0.5)
        original_results.append(result)
        print(f"\nOriginal result for batch {i} (lag={lags[i]}):\n{result}")
    original_output = torch.stack(original_results)

    # ベクトル化された実装（固定重み）
    vectorized_output = slide_sum_seq_torch_vectorized(X, Y, lags, base_ratio_start=0.5, base_ratio_end=0.5)
    print(f"\nVectorized output:\n{vectorized_output}")

    # 結果の比較
    is_equal = torch.allclose(original_output, vectorized_output, atol=1e-6)
    max_diff = torch.max(torch.abs(original_output - vectorized_output))

    print(f"\nTest 1 - Single X tensor (fixed base_ratio_start=0.5, base_ratio_end=0.5):")
    print(f"Results are equal: {is_equal}")
    print(f"Maximum difference: {max_diff}")

    if not is_equal:
        print(f"Original output:\n{original_output}")
        print(f"Vectorized output:\n{vectorized_output}")
        print(f"Difference:\n{original_output - vectorized_output}")

    # テストケース2: バッチ化されたXテンソル（固定重み）
    X_batch = X.unsqueeze(0).expand(num_batches, -1, -1)
    vectorized_output2 = slide_sum_seq_torch_vectorized(X_batch, Y, lags, base_ratio_start=0.5, base_ratio_end=0.5)

    is_equal2 = torch.allclose(original_output, vectorized_output2, atol=1e-6)
    max_diff2 = torch.max(torch.abs(original_output - vectorized_output2))

    print(f"\nTest 2 - Batched X tensor (fixed base_ratio_start=0.5, base_ratio_end=0.5):")
    print(f"Results are equal: {is_equal2}")
    print(f"Maximum difference: {max_diff2}")

    # --- 線形重みのテスト ---
    base_ratio_start = 0.2
    base_ratio_end = 0.8
    print("\n==== Test 3: 線形重み (base_ratio_start=0.2, base_ratio_end=0.8) ====")
    # 元の実装で各バッチを処理（線形重み）
    original_results_linear = []
    for i in range(num_batches):
        result = slide_sum_seq_torch(X, Y[i], lags[i].item(), base_ratio_start=base_ratio_start, base_ratio_end=base_ratio_end)
        original_results_linear.append(result)
        print(f"\nOriginal result (linear) for batch {i} (lag={lags[i]}):\n{result}")
    original_output_linear = torch.stack(original_results_linear)

    # ベクトル化された実装（線形重み）
    vectorized_output_linear = slide_sum_seq_torch_vectorized(X, Y, lags, base_ratio_start=base_ratio_start, base_ratio_end=base_ratio_end)
    print(f"\nVectorized output (linear):\n{vectorized_output_linear}")

    # 結果の比較
    is_equal_linear = torch.allclose(original_output_linear, vectorized_output_linear, atol=1e-6)
    max_diff_linear = torch.max(torch.abs(original_output_linear - vectorized_output_linear))

    print(f"\nTest 3 - Single X tensor (linear base_ratio_start=0.2, base_ratio_end=0.8):")
    print(f"Results are equal: {is_equal_linear}")
    print(f"Maximum difference: {max_diff_linear}")

    if not is_equal_linear:
        print(f"Original output (linear):\n{original_output_linear}")
        print(f"Vectorized output (linear):\n{vectorized_output_linear}")
        print(f"Difference (linear):\n{original_output_linear - vectorized_output_linear}")

    return is_equal and is_equal2 and is_equal_linear


# テストを実行（コメントアウト）
# if __name__ == "__main__":
    # test_vectorized_implementation()
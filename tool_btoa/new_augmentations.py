import numpy as np
import torch


def gen_new_aug(sample, args, DEVICE):
    """
    基本的なフーリエ領域でのMixup手法
    - 単一のランダムな混合係数を使用
    - 振幅のみを混合し、位相は元のサンプルのまま保持
    - 最もシンプルな実装

    Args:
        sample: 入力時系列データ [batch_size, seq_len, features]
        args: 設定パラメータ
        DEVICE: 使用デバイス

    Returns:
        mixed_samples_time: 混合された時系列データ
    """
    fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
    index = torch.randperm(sample.size(0))
    mixing_coeff = (0.9 - 1) * torch.rand(1) + 1
    abs_fft = torch.abs(fftsamples)
    phase_fft = torch.angle(fftsamples)
    mixed_abs = abs_fft * mixing_coeff + (1 - mixing_coeff) * abs_fft[index]
    z =  torch.polar(mixed_abs, phase_fft) # Go back to fft
    mixed_samples_time = torch.fft.irfft(z, dim=1, norm='ortho')
    return mixed_samples_time

def gen_new_aug_2(sample, args, inds, out, similarities):
    """
    類似度ベースの高度なフーリエ領域Mixup手法
    - 類似度に基づいて混合係数を決定
    - 振幅と位相の両方を混合
    - 複数のサンプルを生成（mini_batch分）

    Args:
        sample: 入力時系列データ
        args: 設定パラメータ（mini_batch等を含む）
        inds: インデックス
        out: 出力
        similarities: サンプル間の類似度行列

    Returns:
        mixed_samples_time: 混合された時系列データ（複数サンプル）
    """
    fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
    similarities = similarities[inds]
    inds = torch.arange(sample.size(0))
    # 類似度に基づいて混合係数を設定
    mixing_coeff = mixing_coefficient_set_for_each(similarities, inds, args)

    coeffs = mixing_coeff.squeeze()
    new_tensor = torch.tensor([1.0])
    coeffs = torch.cat((new_tensor,coeffs),dim=0).to(sample.device)
    abs_fft = torch.abs(fftsamples).to(sample.device)
    phase_fft = torch.angle(fftsamples).to(sample.device)

    fix_x = abs_fft[0].unsqueeze(0).repeat(args.mini_batch+1, 1, 1).to(sample.device)
    #print(fix_xabs_fft.device(),len(coeffs),len(inds))
    mixed_abs = fix_x * coeffs[:, None, None] + (1 - coeffs[:, None, None]) * abs_fft[inds]

    mixed_phase = phase_mix(phase_fft,args, inds, similarities)

    z =  torch.polar(mixed_abs, mixed_phase)
    mixed_samples_time = torch.fft.irfft(z, dim=1, norm='ortho')
    return mixed_samples_time




def gen_new_aug_test(sample, args, inds, out, similarities):
    """
    gen_new_aug_2のテスト版
    - gen_new_aug_2とほぼ同じだが、ランダムなインデックスを使用
    - デバッグやテスト用の実装

    Args:
        sample: 入力時系列データ
        args: 設定パラメータ
        inds: インデックス（使用されない）
        out: 出力（使用されない）
        similarities: サンプル間の類似度行列

    Returns:
        mixed_samples_time: 混合された時系列データ
    """
    fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
    inds = torch.randperm(sample.size(0))
    mixing_coeff = mixing_coefficient_set_for_each(similarities, inds, args)
    coeffs = mixing_coeff.squeeze()

    abs_fft = torch.abs(fftsamples)
    phase_fft = torch.angle(fftsamples)
    mixed_abs = abs_fft * coeffs[:, None, None] + (1 - coeffs[:, None, None]) * abs_fft[inds]
    mixed_phase = phase_mix(phase_fft, inds, similarities)
    z =  torch.polar(mixed_abs, mixed_phase)
    mixed_samples_time = torch.fft.irfft(z, dim=1, norm='ortho')
    return mixed_samples_time




def gen_new_aug_3_ablation(sample, args, DEVICE, similarities):
    """
    アブレーション研究用：ランダム係数を使用したフーリエ領域Mixup
    - 類似度ベースではなく、ランダムな混合係数を使用
    - 振幅と位相の両方を混合
    - 提案手法の有効性を検証するための比較実験用

    Args:
        sample: 入力時系列データ
        args: 設定パラメータ
        DEVICE: 使用デバイス
        similarities: 類似度（使用されない）

    Returns:
        mixed_samples_time: 混合された時系列データ
    """
    fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
    inds = torch.randperm(sample.size(0))
    coeffs = torch.ones(sample.shape[0])
    coeffs = torch.nn.init.trunc_normal_(coeffs,1,0.1,0.9,1)

    abs_fft = torch.abs(fftsamples)
    phase_fft = torch.angle(fftsamples)
    mixed_abs = abs_fft * coeffs[:, None, None] + (1 - coeffs[:, None, None]) * abs_fft[inds]
    dtheta, sign = phase_mix_2(phase_fft, inds)
    mixed_phase = phase_fft + (1-coeffs[:, None, None]) * torch.abs(dtheta) * sign
    z =  torch.polar(mixed_abs, mixed_phase)
    mixed_samples_time = torch.fft.irfft(z, dim=1, norm='ortho')
    return mixed_samples_time

def gen_new_aug_4_comparison(sample, args, DEVICE):
    """
    比較実験用：ランダム位相変化を適用
    - gen_new_aug_3_ablationとほぼ同じ
    - 振幅は混合するが、位相はランダムに変化させる
    - 位相の重要性を検証するための比較実験用

    Args:
        sample: 入力時系列データ
        args: 設定パラメータ
        DEVICE: 使用デバイス

    Returns:
        mixed_samples_time: 混合された時系列データ
    """
    fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
    inds = torch.randperm(sample.size(0))
    coeffs = torch.ones(sample.shape[0])
    coeffs = torch.nn.init.trunc_normal_(coeffs,1,0.1,0.9,1)

    abs_fft = torch.abs(fftsamples)
    phase_fft = torch.angle(fftsamples)
    mixed_abs = abs_fft * coeffs[:, None, None] + (1 - coeffs[:, None, None]) * abs_fft[inds]
    dtheta, sign = phase_mix_2(phase_fft, inds)
    mixed_phase = phase_fft + (1-coeffs[:, None, None]) * torch.abs(dtheta) * sign
    z =  torch.polar(mixed_abs, mixed_phase)
    mixed_samples_time = torch.fft.irfft(z, dim=1, norm='ortho')
    return mixed_samples_time

def opposite_phase(sample, args, DEVICE, similarities):
    """
    位相補間の重要性を示すための実験
    - gen_new_aug_3_ablationとほぼ同じだが、位相の符号を反転
    - 位相の方向性の重要性を検証

    Args:
        sample: 入力時系列データ
        args: 設定パラメータ
        DEVICE: 使用デバイス
        similarities: 類似度（使用されない）

    Returns:
        mixed_samples_time: 混合された時系列データ
    """
    fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
    inds = torch.randperm(sample.size(0))
    coeffs = torch.ones(sample.shape[0])
    coeffs = torch.nn.init.trunc_normal_(coeffs,1,0.1,0.9,1)

    abs_fft = torch.abs(fftsamples)
    phase_fft = torch.angle(fftsamples)
    mixed_abs = abs_fft * coeffs[:, None, None] + (1 - coeffs[:, None, None]) * abs_fft[inds]
    dtheta, sign = phase_mix_2(phase_fft, inds)
    mixed_phase = phase_fft + (1-coeffs[:, None, None]) * torch.abs(dtheta) * -sign
    z =  torch.polar(mixed_abs, mixed_phase)
    mixed_samples_time = torch.fft.irfft(z, dim=1, norm='ortho')
    return mixed_samples_time

def STAug(sample, args, DEVICE):
    """
    スペクトル・時間領域での拡張手法（比較実験用）
    - EMD（Empirical Mode Decomposition）を使用
    - 各IMF（Intrinsic Mode Function）にランダムな重みを適用
    - フーリエ変換ベースではない別のアプローチ

    Args:
        sample: 入力時系列データ
        args: 設定パラメータ
        DEVICE: 使用デバイス

    Returns:
        拡張された時系列データ
    """
    # GPUテンソルをCPUのnumpy配列に変換
    sample = sample.detach().cpu().numpy()

    # バッチ内の各サンプルに対して処理
    for i in range(sample.shape[0]):
        # 各チャンネル（特徴量）に対して処理
        for k in range(sample.shape[2]):
            # EMDでIMFを抽出
            current_imf = emd.sift.sift(sample[i,:,k])

            # 0から2の範囲でランダムな重みを生成
            w = np.random.uniform(0, 2, current_imf.shape[1])

            # IMFに重みを適用
            weighted_imfs = current_imf * w[None,:]

            # 重み付きIMFを合計して新しい信号を生成
            s_prime = np.sum(weighted_imfs, axis=1)
            sample[i,:,k] = s_prime

    # numpy配列をPyTorchテンソルに戻す
    return torch.from_numpy(sample).float()


def vanilla_mix_up(sample):
    """
    基本的なMixup手法（時間領域）
    - 時間領域で直接サンプルを混合
    - 0.75から1.0の範囲でランダムな混合係数を使用
    - 最もシンプルなMixup実装

    Args:
        sample: 入力時系列データ

    Returns:
        mixed_data: 混合された時系列データ
    """
    # 0.75から1.0の範囲でランダムな混合係数を生成
    mixing_coeff = (0.75 - 1) * torch.rand(1) + 1

    # バッチ内のサンプルをランダムに並び替え
    index = torch.randperm(sample.size(0))

    # 線形結合でデータを混合
    mixed_data = mixing_coeff * sample + (1 - mixing_coeff) * sample[index]
    return mixed_data


def vanilla_mix_up_geo(sample):
    """
    幾何学的Mixup手法（時間領域）
    - 時間領域で幾何学的平均を使用してサンプルを混合
    - 0.7から1.0の範囲でランダムな混合係数を使用
    - 線形結合ではなく幾何学的結合

    Args:
        sample: 入力時系列データ

    Returns:
        mixed_data: 混合された時系列データ
    """
    # 0.7から1.0の範囲でランダムな混合係数を生成
    mixing_coeff = (0.7 - 1) * torch.rand(1) + 1

    # バッチ内のサンプルをランダムに並び替え
    index = torch.randperm(sample.size(0))

    # 幾何学的結合でデータを混合
    mixed_data = sample**mixing_coeff * sample[index]**(1 - mixing_coeff)
    return mixed_data


def vanilla_mix_up_binary(sample):
    """
    バイナリMixup手法（時間領域）
    - バイナリマスクを使用してサンプルを混合
    - 各要素に対してランダムなマスクを生成
    - より細かい制御が可能

    Args:
        sample: 入力時系列データ

    Returns:
        x_mixup: 混合された時系列データ
    """
    alpha = 0.8
    # alphaから1の範囲でランダムな値を生成
    lam = torch.empty(sample.shape).uniform_(alpha, 1)

    # ランダムなバイナリマスクを生成
    mask = torch.empty(sample.shape).bernoulli_(lam)

    # シャッフルされたサンプルを取得
    x_shuffle = sample[torch.randperm(sample.shape[0])]

    # マスクを使用してサンプルを混合
    x_mixup = sample * mask + x_shuffle * (1 - mask)
    return x_mixup


def best_mix_up(sample, args, similarities, DEVICE):
    """
    類似度ベースのMixup手法（時間領域）
    - 類似度に基づいて混合係数を決定
    - 時間領域で線形結合を適用
    - フーリエ変換を使用しないアブレーション実験用

    Args:
        sample: 入力時系列データ
        args: 設定パラメータ
        similarities: サンプル間の類似度行列
        DEVICE: 使用デバイス

    Returns:
        mixed_data: 混合された時系列データ
    """
    # バッチ内のサンプルをランダムに並び替え
    index = torch.randperm(sample.size(0))

    # 類似度に基づいて混合係数を計算
    coeffs = mixing_coefficient_set_for_each(similarities, index, args)
    coeffs = coeffs.squeeze()

    # 時間領域で線形結合を適用
    mixed_data = coeffs[:, None, None] * sample + (1 - coeffs[:, None, None]) * sample[index]
    return mixed_data


def mixing_coefficient_set_for_each(similarities, inds, args):
    """
    類似度に基づいて混合係数を設定する関数
    - 類似度が閾値（0.8）を超える場合はランダムな係数を使用
    - 類似度が閾値以下の場合は切断正規分布で係数を生成
    - 類似度ベースの混合の核心部分

    Args:
        similarities: サンプル間の類似度行列
        inds: インデックス
        args: 設定パラメータ（mean, std, low_limit, high_limit）

    Returns:
        distances: 計算された混合係数
    """
    threshold = 0.8  # 類似度の閾値

    # 初期化
    mixing_coefficient = torch.ones(similarities.shape)
    similarities = similarities.cpu()
    distances = similarities
    mixing_coefficient = torch.ones(similarities.shape)

    # 類似度が閾値を超える場合は0.7から1.0の範囲でランダムな係数を使用
    distances[distances > threshold] = (0.7 - 1) * torch.rand(1) + 1

    # 切断正規分布で係数を生成
    mixing_coefficient = torch.ones(distances.shape)
    mixing_coefficient = torch.nn.init.trunc_normal_(
        mixing_coefficient, args.mean, args.std, args.low_limit, args.high_limit
    )

    # 類似度が閾値以下の場合は生成された係数を使用
    distances[distances <= threshold] = mixing_coefficient[distances <= threshold]

    return distances


def phase_mix(phase_fft, args, inds, similarities):
    """
    位相の混合を行う関数
    - 類似度に基づいて位相の混合係数を決定
    - 位相の差分を計算して適切に混合
    - 複数のサンプルを生成（mini_batch分）

    Args:
        phase_fft: フーリエ変換された位相
        args: 設定パラメータ
        inds: インデックス
        similarities: 類似度行列

    Returns:
        mixed_phase: 混合された位相
    """
    # 元の位相を複製
    fix_pahse = phase_fft[0].unsqueeze(0).repeat(args.mini_batch+1, 1, 1)

    # 位相の差分を計算
    phase_difference = fix_pahse - phase_fft[inds]
    dtheta = phase_difference % (2 * torch.pi)

    # 位相の差分を-πからπの範囲に正規化
    dtheta[dtheta > torch.pi] -= 2 * torch.pi
    clockwise = dtheta > 0
    sign = torch.where(clockwise, -1, 1)

    # 類似度に基づいて位相の混合係数を計算
    coeffs = torch.squeeze(mixing_coefficient_set_for_each_phase(similarities, inds))
    new_tensor = torch.tensor([1.0])
    coeffs = torch.cat((new_tensor, coeffs), dim=0).to(phase_fft.device)

    # 位相を混合
    mixed_phase = phase_fft
    mixed_phase = fix_pahse.to(phase_fft.device) + (1-coeffs[:, None, None]) * torch.abs(dtheta) * sign
    return mixed_phase


def phase_mix_2(phase_fft, inds):
    """
    位相の差分と符号を計算する関数
    - 位相の差分を計算して適切な符号を決定
    - よりシンプルな位相混合の実装

    Args:
        phase_fft: フーリエ変換された位相
        inds: インデックス

    Returns:
        dtheta: 位相の差分
        sign: 符号
    """
    # 位相の差分を計算
    phase_difference = phase_fft - phase_fft[inds]
    dtheta = phase_difference % (2 * torch.pi)

    # 位相の差分を-πからπの範囲に正規化
    dtheta[dtheta > torch.pi] -= 2 * torch.pi
    clockwise = dtheta > 0
    locs = torch.where(torch.abs(phase_difference) > torch.pi, -1, 1)
    sign = torch.where(clockwise, -1, 1)
    return dtheta, sign

def mixing_coefficient_set_for_each_phase(similarities, inds):
    threshold = 0.8
    mixing_coefficient = torch.ones(similarities.shape)
    similarities = similarities.cpu()
    distances = similarities

    mixing_coefficient = torch.ones(similarities.shape)
    distances[distances>threshold] = (0.9 - 1) * torch.rand(1) + 1
    mixing_coefficient = torch.ones(distances.shape)
    mixing_coefficient = torch.nn.init.trunc_normal_(mixing_coefficient,1,0.1,0.9,1)
    # mixing_coefficient = torch.nn.init.trunc_normal_(mixing_coefficient,0.9,0.2,0.7,1)
    distances[distances<=threshold] = mixing_coefficient[distances<=threshold]
    #distances = torch.from_numpy(distances)
    return distances

def check_max_not_selected(max_indices, indices, abs_fft):
    for i in range(len(max_indices)):
        while indices[i] == max_indices[i].item():
            #np.random.shuffle(indices)
            indices = np.random.choice(np.ceil(abs_fft.size(1)/2).astype(int),abs_fft.size(2))
    return indices


######################################### For Supervised Learning Paradigm #########################################

def vanilla_mixup_sup(sample, target, alpha=0.3):
    size_of_batch = sample.size(0)
    # Choose quarters of the batch to mix
    indices = torch.randperm(size_of_batch)
    m = torch.distributions.beta.Beta(torch.tensor([alpha]), torch.tensor([alpha]))
    mixing_coeff = m.sample()
    # Mix the data
    mixed_data = mixing_coeff * sample + (1 - mixing_coeff) * sample[indices]
    return mixed_data, target, mixing_coeff, target[indices]


def gen_new_aug_3_ablation_sup(sample, args, DEVICE, target, alpha=0.2):
    fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
    inds = torch.randperm(sample.size(0))
    m = torch.distributions.beta.Beta(torch.tensor([alpha]), torch.tensor([alpha]))
    coeffs = m.sample()

    abs_fft = torch.abs(fftsamples)
    phase_fft = torch.angle(fftsamples)
    mixed_abs = abs_fft * coeffs + (1 - coeffs) * abs_fft[inds]

    dtheta, sign = phase_mix_2(phase_fft, inds)
    dtheta2, sign2 = phase_mix_2(phase_fft[inds], torch.linspace(0,63,64,dtype=inds.dtype))
    #mixed_phase = phase_fft if coeffs > 0.5 else phase_fft[inds]
    phase_coeff = (0.9 - 1) * torch.rand(1) + 1
    mixed_phase = phase_fft + (1-phase_coeff) * torch.abs(dtheta) * sign if coeffs > 0.5 else phase_fft[inds] + (1-phase_coeff) * torch.abs(dtheta2) * sign2
    z =  torch.polar(mixed_abs, mixed_phase)
    mixed_samples_time = torch.fft.irfft(z, dim=1, norm='ortho')
    return mixed_samples_time, target, coeffs, target[inds]

def cutmix_sup(data, target, alpha=1.):
    batch_size = data.size(0)
    indices = torch.randperm(batch_size)
    shuffled_data = data[indices]

    m = torch.distributions.beta.Beta(torch.tensor([alpha]), torch.tensor([alpha]))
    lam = m.sample()

    cut_len = int(lam * data.size(1))
    cut_start = np.random.randint(0, data.size(1) - cut_len + 1)

    data[:, cut_start:cut_start+cut_len] = shuffled_data[:, cut_start:cut_start+cut_len]
    return data, target, lam, target[indices]

def binary_mixup_sup(sample, target, alpha=0.2):
    lam = torch.empty(sample.shape).uniform_(alpha, 1)
    mask = torch.empty(sample.shape).bernoulli_(lam)
    indices = torch.randperm(sample.shape[0])
    x_shuffle = sample[indices]
    x_mixup = sample * mask + x_shuffle * (1 - mask)
    return x_mixup, target, lam, target[indices]


def gen_new_aug_2_sup(sample, args, inds, out, DEVICE, similarities, target):
    fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
    inds = torch.randperm(sample.size(0))
    mixing_coeff = mixing_coefficient_set_for_each(similarities, inds, args)
    coeffs = mixing_coeff.squeeze()

    abs_fft = torch.abs(fftsamples)
    phase_fft = torch.angle(fftsamples)
    mixed_abs = abs_fft * coeffs[:, None, None] + (1 - coeffs[:, None, None]) * abs_fft[inds]
    mixed_phase = phase_mix(phase_fft, inds, similarities)
    #z =  torch.polar(mixed_abs, torch.angle(fftsamples)) # Go back to fft
    z =  torch.polar(mixed_abs, mixed_phase)
    mixed_samples_time = torch.fft.irfft(z, dim=1, norm='ortho')
    return mixed_samples_time, target, coeffs, target[inds]


def mag_mixup_sup(sample, args, DEVICE, target, alpha=0.2):
    fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
    index = torch.randperm(sample.size(0))
    m = torch.distributions.beta.Beta(torch.tensor([alpha]), torch.tensor([alpha]))
    coeffs = m.sample()
    abs_fft = torch.abs(fftsamples)
    phase_fft, phase_fft2 = torch.angle(fftsamples), torch.angle(fftsamples[index])
    mixed_abs = abs_fft * coeffs + (1 - coeffs) * abs_fft[index]
    z =  torch.polar(mixed_abs, phase_fft) if coeffs > 0.5 else torch.polar(mixed_abs, phase_fft2)
    mixed_samples_time = torch.fft.irfft(z, dim=1, norm='ortho')
    #value = torch.roll(value,5,1)
    return mixed_samples_time, target, coeffs, target[index]
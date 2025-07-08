from data_provider.data_loader import *
from torch.utils.data import DataLoader, DistributedSampler

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'ETTh1_CI': Dataset_ETT_hour_CI,
    'ETTh2_CI': Dataset_ETT_hour_CI,
    'ETTm1_CI': Dataset_ETT_minute_CI,
    'ETTm2_CI': Dataset_ETT_minute_CI,
    'custom_CI': Dataset_Custom_CI,
}

flag2num = {'train': 0, 'val': 1, 'test': 2, 'pred': 3}

# =====================================
# データセット・データローダ生成の工場関数群
# =====================================

def get_dataset(args, flag, device='cpu', wrap_class=None, borders=None, take_post=0, take_pre=False, noise=0, **kwargs):
    """
    指定された引数・フラグに基づき、適切なデータセットインスタンスを生成する関数。
    wrap_classでデータセットのラッピングも可能。

    Parameters:
        args: 実験やデータセットの各種設定を持つNamespace。
        flag: 'train', 'val', 'test'のデータ分割の指定。
        device: データを配置するデバイス。
        wrap_class: データセットをラップするクラスまたはクラスのリスト。
            例：wrap_class=[Dataset_Recent] のように指定すると、
            生成したデータセットをDataset_Recentなどでラップし、
            追加の前処理やバッチ生成ロジックを付与できる。
            オンライン学習や特殊なデータ分割・加工が必要な場合に利用。
        borders: データ分割の境界設定
        take_post: 後方に追加で取得するデータポイント数
        take_pre: 前方に追加で取得するデータポイント数
        noise: ノイズの強度（0の場合はノイズなし）
        その他: データセット生成に必要な追加パラメータ。

    Returns:
        データセットインスタンス
    """
    # =====================================
    # 1. 時間エンコーディング設定の初期化
    # =====================================
    # defaultではtimeenc=2になる
    if not hasattr(args, 'timeenc'):
        # timeencが設定されていない場合、embed設定に基づいて自動設定
        # embed='timeF'の場合は時間特徴量を使用（timeenc=1）、それ以外は使用しない（timeenc=0）
        args.timeenc = 0 if not hasattr(args, 'embed') or args.embed != 'timeF' else 1

    # =====================================
    # 2. ラッパークラスの正規化
    # =====================================
    # online_methodの場合，wrap_classが指定される(Dataset_Recent)
    if wrap_class is not None:
        if not isinstance(wrap_class, list):
            # 単一クラスの場合はリストに変換
            wrap_class = [wrap_class]

    # =====================================
    # 3. データ境界（border）の設定
    # =====================================
    border = None
    if borders is not None:
        if flag == 'pred' and len(borders[0]) == 3:
            # 予測フラグで3分割の場合、訓練データの範囲を使用
            flag = 'train'
            border = (borders[0][flag2num['train']], borders[1][flag2num['test']])
        else:
            # 通常の場合、指定されたフラグに対応する境界を使用
            border = (borders[0][flag2num[flag]], borders[1][flag2num[flag]])

        # 訓練以外のフラグ（val, test）の場合の特別処理
        if flag != 'train':
            if Dataset_Recent in wrap_class or take_pre > 0:
                # Dataset_Recentが使用される場合やtake_preが設定されている場合
                if take_pre > 1:
                    # take_preが1より大きい場合、その分だけ前方に拡張
                    start = border[0] - take_pre
                else:
                    # デフォルトでは予測長分だけ前方に拡張（オンライン学習用）
                    start = border[0] - args.pred_len
                # 開始位置が0未満にならないように調整
                border = (max(0, start), border[1])

        # take_postが設定されている場合、後方に拡張
        if take_post > 0:
            border = (border[0], border[1] + take_post)

    # =====================================
    # 4. データセットインスタンスの生成
    # =====================================
    # data_dictから適切なデータセットクラスを選択してインスタンス化
    data_set = data_dict[args.data](
        root_path=args.root_path,      # データのルートパス
        data_path=args.data_path,      # データファイルのパス
        flag=flag,                     # データ分割フラグ
        size=[args.seq_len, args.label_len, args.pred_len],  # シーケンス長、ラベル長、予測長
        features=args.features,        # 特徴量設定（'M', 'S', 'MS'）
        target=args.target,            # ターゲット変数名
        timeenc=args.timeenc,          # 時間エンコーディング設定
        freq=args.freq,                # 時間頻度（'h', 'd', 'm'など）
        train_only=args.train_only,    # 訓練のみフラグ
        border=border,                 # データ境界
        borders=args.borders if hasattr(args, 'borders') else None,  # 全境界設定
        ratio=args.ratio if hasattr(args, 'ratio') else None         # データ比率
    )

    # =====================================
    # 5. GPU固定（pin_gpu）の処理
    # =====================================
    if args.pin_gpu and hasattr(data_set, 'data_x'):
        # GPU固定が有効で、データセットにdata_x属性がある場合
        # データをGPUに移動して固定
        data_set.data_x = torch.tensor(data_set.data_x, dtype=torch.float32, device=device)
        data_set.data_y = torch.tensor(data_set.data_y, dtype=torch.float32, device=device)

        # 時間特徴量が必要なモデルの場合、data_stampもGPUに移動
        from settings import need_x_mark, need_x_y_mark
        if args.model in need_x_mark or args.model in need_x_y_mark or args.use_time or \
                hasattr(args, 'online_method') and args.online_method == 'OneNet':
            data_set.data_stamp = torch.tensor(data_set.data_stamp, dtype=torch.float32, device=device)

    # =====================================
    # 6. ノイズの追加（オプション）
    # =====================================
    if noise:
        print("Modify time series with strength =", noise)
        # 時系列データにノイズを追加
        # 過去の値との差分に基づいてノイズを生成
        for i in range(3, len(data_set.data_y)):
            # 入力データ（data_x）にノイズを追加
            data_set.data_x[i] += 0.01 * (i // noise) * (data_set.data_x[i-1] - data_set.data_x[i-2])
            # 出力データ（data_y）にノイズを追加
            data_set.data_y[i] += 0.01 * (i // noise) * (data_set.data_y[i-1] - data_set.data_y[i-2])

    # =====================================
    # 7. ラッパークラスの適用
    # =====================================
    if wrap_class is not None:
        # 指定されたラッパークラスを順次適用
        for cls in wrap_class:
            data_set = cls(data_set, **kwargs)

    # =====================================
    # 8. 結果の出力と返却
    # =====================================
    print(flag, len(data_set))  # フラグとデータセットの長さを出力
    return data_set


def get_dataloader(data_set, args, flag, sampler=None):
    """
    データセットに対応するDataLoaderを生成する関数。
    バッチサイズやシャッフル設定はargsとflagで制御。
    """
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    elif flag == 'online':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag and args.local_rank == -1,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=False,
        sampler=sampler if args.local_rank == -1 or flag in ['online', 'test'] else DistributedSampler(data_set))
    return data_loader


def data_provider(args, flag, device='cpu', wrap_class=None, sampler=None, **kwargs):
    """
    データセットとデータローダを同時に返すユーティリティ関数。
    flagは'train', 'val', 'test'のいずれか
    exp_basicでは_get_dataから，data_providerが呼ばれる．
    exp_onlineではonlineから，get_dataset(flag='test')の後にget_dataloader(flag='online')が呼ばれる．
    """
    data_set = get_dataset(args, flag, device, wrap_class=wrap_class, **kwargs)
    data_loader = get_dataloader(data_set, args, flag, sampler)
    return data_set, data_loader

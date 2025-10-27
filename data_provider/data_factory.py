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

# =====================================
# Factory functions for dataset and dataloader generation
# =====================================

def get_dataset(args, flag, setting, device='cpu', wrap_class=None, noise=0, **kwargs):
    """
    Function to generate appropriate dataset instance based on specified arguments and flags.
    Dataset wrapping is also possible with wrap_class.

    Parameters:
        args: Namespace with various experiment and dataset settings.
        flag: Data split specification: 'train', 'valid', 'test'.
        setting: Dataset setting: 'online', 'offline_learn', 'offline_test'.
        device: Device to place data on.
        wrap_class: Class or list of classes to wrap dataset.
            Example: Specifying wrap_class=[Dataset_Recent] wraps
            generated dataset with Dataset_Recent etc.,
            adding additional preprocessing or batch generation logic.
            Used when online learning or special data splitting/processing is needed.
        noise: Noise intensity (no noise if 0)
        Other: Additional parameters needed for dataset generation.

    Returns:
        Dataset instance
    """
    # =====================================
    # 1. Initialize time encoding settings
    # =====================================
    # By default, timeenc=2
    if not hasattr(args, 'timeenc'):
        # If timeenc is not set, auto-configure based on embed setting
        # Use time features if embed='timeF' (timeenc=1), otherwise don't use (timeenc=0)
        args.timeenc = 0 if not hasattr(args, 'embed') or args.embed != 'timeF' else 1

    # =====================================
    # 2. Normalize wrapper class
    # =====================================
    # For online_method, wrap_class is specified (Dataset_Recent)
    if wrap_class is not None:
        if not isinstance(wrap_class, list):
            # Convert to list if single class
            wrap_class = [wrap_class]

    # =====================================
    # 3. Generate dataset instance
    # =====================================

    # Select appropriate dataset class from data_dict and instantiate
    data_set = data_dict[args.data](
        root_path=args.root_path,      # Data root path
        data_path=args.data_path,      # Data file path
        flag=flag,                     # Data split flag (train, valid, test)
        setting=setting,               # Dataset setting (online, offline_learn, offline_test)
        size=[args.seq_len, args.label_len, args.pred_len],  # Sequence length, label length, prediction length
        features=args.features,        # Feature setting ('M', 'S', 'MS')
        target=args.target,            # Target variable name
        timeenc=args.timeenc,          # Time encoding setting
        freq=args.freq,                # Time frequency ('h', 'd', 'm', etc.)
        border_type=args.border_type,  # Type of border setting
    )

    # =====================================
    # 4. GPU pinning (pin_gpu) processing
    # =====================================
    if args.pin_gpu and hasattr(data_set, 'data_x'):
        # If GPU pinning is enabled and dataset has data_x attribute
        # Move data to GPU and pin
        data_set.data_x = torch.tensor(data_set.data_x, dtype=torch.float32, device=device)
        data_set.data_y = torch.tensor(data_set.data_y, dtype=torch.float32, device=device)

        # If model needs time features, also move data_stamp to GPU
        from settings import need_x_mark, need_x_y_mark
        if args.model in need_x_mark or args.model in need_x_y_mark or args.use_time or \
                hasattr(args, 'online_method') and args.online_method == 'OneNet':
            data_set.data_stamp = torch.tensor(data_set.data_stamp, dtype=torch.float32, device=device)

    # =====================================
    # 5. Add noise (optional)
    # =====================================
    if noise:
        print("Modify time series with strength =", noise)
        # Add noise to time series data
        # Generate noise based on difference from past values
        for i in range(3, len(data_set.data_y)):
            # Add noise to input data (data_x)
            data_set.data_x[i] += 0.01 * (i // noise) * (data_set.data_x[i-1] - data_set.data_x[i-2])
            # Add noise to output data (data_y)
            data_set.data_y[i] += 0.01 * (i // noise) * (data_set.data_y[i-1] - data_set.data_y[i-2])

    # =====================================
    # 6. Apply wrapper classes
    # =====================================
    if wrap_class is not None:
        # Apply specified wrapper classes sequentially
        for cls in wrap_class:
            data_set = cls(data_set, **kwargs)

    # =====================================
    # 7. Output and return result
    # =====================================
    print(flag, setting, len(data_set))  # Output flag and dataset length
    return data_set


def get_dataloader(data_set, args, setting, sampler=None):
    """
    Function to generate DataLoader corresponding to dataset.
    Batch size and shuffle settings are controlled by args and flag.
    """
    if setting == 'offline_learn':
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
    elif setting == 'offline_test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    elif setting == 'online':
        shuffle_flag = False
        drop_last = False
        batch_size = 1

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag and args.local_rank == -1,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=False,
        sampler=sampler if args.local_rank == -1 or setting in ['online', 'offline_test'] else DistributedSampler(data_set))
    return data_loader

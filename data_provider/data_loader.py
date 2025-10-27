# -*- coding: utf-8 -*-
from typing import Union

import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from util.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


def get_alldata(filename='electricity.csv', root_path='./'):
    """
    Function to load time series data from specified filename and root path,
    and return it as pandas.DataFrame.
    - Supports multiple formats including CSV, HDF5, NPZ, text, etc.
    - Generates and formats date column according to filename (e.g., wind data, nyc data).
    - Finally formats so 'date' column is at the front.
    """
    path = os.path.join(root_path, filename)
    if filename.endswith('.csv'):
        if filename in ['NOAA.csv', 'Powersupply.csv', 'AU_Electricity.csv']:
            df = pd.read_csv(path, header=None)
            df = df.drop(columns=df.columns[-1])
            if filename.startswith('NOAA'):
                df['date'] = pd.date_range(start='2016-01-01', periods=len(df), freq='D')
            elif filename.startswith('Powersupply'):
                df['date'] = pd.date_range(start='1995-01-01', periods=len(df), freq='H')
            elif filename.startswith('AU_Electricity'):
                df['date'] = pd.date_range(start='2016-01-01', periods=len(df), freq='5min')
        elif filename=='electricityselect.csv':
            filename = 'electricity.csv'
            path = os.path.join(root_path, filename)
            df = pd.read_csv(path)
            select_cols = ['date', '46', '121', '310', '118', '119', '117', '178', '217', '147', '225']
            df = df[select_cols]
        elif filename=='trafficselect.csv':
            filename = 'traffic.csv'
            path = os.path.join(root_path, filename)
            df = pd.read_csv(path)
            select_cols = ['date', '448', '496', '353', '664', '445', '833', '571', '565', '106', '44']
            df = df[select_cols]
        elif filename=='ETTh2Select.csv':
            filename = 'ETTh2.csv'
            path = os.path.join(root_path, filename)
            df = pd.read_csv(path)
            select_cols = ['date', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'OT']
            df = df[select_cols]
        else:
            df = pd.read_csv(path)
            if filename.startswith('wind'):
                df['date'] = pd.date_range(start='2000-01-01', periods=len(df), freq='H')
            elif filename in ['nsw_elc_price.csv', 'nsw_elc_demand.csv', 'synthetic_A3.csv', 'synthetic_B3.csv'] :
                df['date'] = pd.date_range(start='2016-01-01', periods=len(df), freq='5min')
    else:
        if filename.startswith('nyc'):
            import h5py
            x = h5py.File(path, 'r')
            data = list()
            for key in x.keys():
                data.append(x[key][:])
            ts = np.stack(data, axis=1)
            ts = ts.reshape((ts.shape[0], ts.shape[1] * ts.shape[2]))
            df = pd.DataFrame(ts)
            df['date'] = pd.date_range(start='2007-04-01', periods=len(df), freq='30T')
        elif filename.endswith('.npz'):
            ts = np.load(path)['data'].astype(np.float32)
            ts = ts.reshape((ts.shape[0], ts.shape[1] * ts.shape[2]))
            df = pd.DataFrame(ts)
            if filename == 'PeMSD4':
                df['date'] = pd.date_range(start='2017-07-01', periods=len(df), freq='5T')
            else:
                df['date'] = pd.date_range(start='2012-03-01', periods=len(df), freq='5T')
        elif filename.endswith('.h5'):
            df = pd.read_hdf(path)
            df['date'] = df.index.values
        elif filename.endswith('.txt'):
            df = pd.read_csv(path, header=None)
            df['date'] = pd.date_range(start='1/1/2007', periods=len(df), freq='10T')
        df = df[[df.columns[-1]] + list(df.columns[:-1])]
    return df

def get_borders_for_border_type(border_type, filename):
    """
    Function to set data boundaries for online learning

    Args:
        border_type: Type of boundary setting
        filename: Data filename

    Returns:
        None: Sets boundary values in args.borders
    """
    borders, ratio = None, None


    if border_type == 'offline':
        flag2num = {'train': 0, 'valid': 1, 'test': 2}
        if filename.startswith('ETTh'):
            # Border settings for ETTh dataset (monthly units)
            border1s = [0, 12*30*24, 16*30*24] # Start points for (train, val, test)
            border2s = [12*30*24, 16*30*24, 20*30*24] # End points for (train, val, test)
            borders = (border1s, border2s)
        elif filename.startswith('ETTm'):
            # Border settings for ETTm dataset (15-minute intervals, monthly units)
            border1s = [0, 12*30*24*4, 16*30*24*4]
            border2s = [12*30*24*4, 16*30*24*4, 20*30*24*4]
            borders = (border1s, border2s)
        else:
            # Other datasets set by ratio
            ratio = (0.6, 0.2, 0.2) # Ratio of training:validation:test data

    elif border_type == 'online':
        flag2num = {'train': 0, 'valid': 1, 'test': 2}
        if filename.startswith('ETTh'):
            border1s = [0, 4*30*24, 5*30*24]
            border2s = [4*30*24, 5*30*24, 20*30*24]
            borders = (border1s, border2s)
        elif filename.startswith('ETTm'):
            border1s = [0, 4*30*24*4, 5*30*24*4]
            border2s = [4*30*24*4, 5*30*24*4, 20*30*24*4]
            borders = (border1s, border2s)
        else:
            ratio = (0.2, 0.05, 0.75)

    elif border_type == 'online3test':
        flag2num = {'train': 0, 'valid': 1, 'test1': 2, 'test2': 3, 'test3': 4}
        if filename.startswith('ETTh'):
            border1s = [0, 4*30*24, 5*30*24, 10*30*24, 15*30*24]
            border2s = [4*30*24, 5*30*24, 10*30*24, 15*30*24, 20*30*24]
            borders = (border1s, border2s)
        elif filename.startswith('ETTm'):
            border1s = [0, 4*30*24*4, 5*30*24*4, 10*30*24*4, 15*30*24*4]
            border2s = [4*30*24*4, 5*30*24*4, 10*30*24*4, 15*30*24*4, 20*30*24*4]
            borders = (border1s, border2s)
        else:
            ratio = (0.2, 0.05, 0.25, 0.25, 0.25)

    return borders, ratio, flag2num

def get_borders_from_ratio(df_length, ratio):
    """
    Calculate boundaries from data length and ratio (supports arbitrary number of phases)

    Args:
        df_length (int): Length of dataframe
        ratio (list): Ratio of each phase [train_ratio, val_ratio, test1_ratio, test2_ratio, ...]

    Returns:
        tuple: (border1s, border2s) - start and end positions of each phase

    Example:
        ratio = [0.2, 0.05, 0.75]  # train, val, test
        ratio = [0.2, 0.05, 0.25, 0.25, 0.25]  # train, val, test1, test2, test3
    """
    if len(ratio) < 2:
        raise ValueError(f"Ratio must have at least 2 elements, got {len(ratio)}")

    # Calculate length of each phase
    phase_lengths = []
    for i, ratio_val in enumerate(ratio):
        if i == 1:  # Calculate remaining for val phase
            continue
        phase_lengths.append(int(df_length * ratio_val))

    # Calculate val phase length (remaining data)
    used_length = sum(phase_lengths)
    val_length = df_length - used_length
    phase_lengths.insert(1, val_length)  # Insert val at second position

    # Calculate boundaries
    border1s = [0]
    border2s = []
    cumulative = 0

    for length in phase_lengths:
        cumulative += length
        border1s.append(cumulative)
        border2s.append(cumulative)

    # Last boundary ends at df_length
    border2s[-1] = df_length

    return (border1s, border2s)

def get_border_for_flag(borders, flag, setting, flag2num, seq_len, pred_len, df_length):
    # =====================================
    # Data border settings; in online learning, train:valid:test boundaries are continuous
    # =====================================
    assert flag in flag2num.keys()
    # Get boundary corresponding to specified flag
    border = (borders[0][flag2num[flag]], borders[1][flag2num[flag]])

    if setting in ['online', 'offline_learn']:
        # Extend forward by input sequence length and prediction length
        start = border[0] - (seq_len + pred_len) + 1
    elif setting in ['offline_test']:
        # Extend forward by input sequence length only
        start = border[0] - seq_len + 1

    if setting in ['online', 'offline_test']:
        # Extend backward by prediction length
        end = border[1] + pred_len
    elif setting in ['offline_learn']:
        end = border[1]

    # Adjust so start position is not less than 0 (becomes 0 for train)
    # Adjust so end position does not exceed data length (becomes data length for last test)
    border = (max(0, start), min(df_length, end))

    return border

def get_borders(border_type, filename, flag, setting, seq_len, pred_len, df_length):
    # Get boundaries according to border type
    borders, ratio, flag2num = get_borders_for_border_type(border_type, filename)
    # If ratio is not None, get boundaries according to ratio
    if borders is None:
        borders = get_borders_from_ratio(df_length, ratio)
    # Get boundaries according to flag
    border1, border2 = get_border_for_flag(borders, flag, setting, flag2num, seq_len, pred_len, df_length)
    return borders, border1, border2


class Dataset_ETT_hour(Dataset):
    """
    PyTorch Dataset class for ETT(hour) dataset.
    - Handles data loading, scaling, time series feature generation,
      train/validation/test split, sample extraction via getitem, etc.
    - Flexible sequence length specification with seq_len, label_len, pred_len.
    - Supports features='S' (univariate) / 'M' (multivariate) / 'MS' (multivariate+target).
    - Returns model input/output/time features via __getitem__.
    """
    def __init__(self, root_path, flag='train', setting='offline_learn', size=None,
                 features='S', data_path='ETTh1.csv', border_type='online',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        self.flag = flag
        self.setting = setting
        self.border_type = border_type

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path # file name
        self.__read_data__()

    def __read_data__(self):
        """
        Internal function to perform data loading, scaling, splitting, and time feature generation.
        - train/val/test boundaries are determined by borders or default values.
        - Data extraction according to features type.
        - Scaling is fit only on training data.
        - Time features are generated by util.timefeatures.time_features.
        """
        self.scaler = StandardScaler()
        df_raw = get_alldata(self.data_path, self.root_path)

        self.borders, border1, border2 = get_borders(self.border_type, self.data_path, self.flag, self.setting, self.seq_len, self.pred_len, len(df_raw))
        border1s, border2s = self.borders

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Get data according to flag
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, self.timeenc, freq=self.freq)
        self.data_stamp = data_stamp.astype(np.float32)

        self.data_x = torch.from_numpy(self.data_x).float()
        self.data_y = torch.from_numpy(self.data_y).float()
        self.data_stamp = torch.from_numpy(self.data_stamp).float()

    def __getitem__(self, index):
        """
        Extract and return sequence data (input, output, time features) from specified index.
        - seq_x: Input sequence
        - seq_y: Prediction target sequence
        - seq_x_mark: Time features of input sequence
        - seq_y_mark: Time features of prediction target sequence
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_end:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        # print("seq_x.shape:",seq_x.shape,"seq_y.shape:",seq_y.shape,"seq_x_mark.shape:",seq_x_mark.shape,"seq_y_mark.shape:",seq_y_mark.shape)
        return seq_x, seq_y, seq_x_mark, seq_y_mark, index

    def __len__(self):
        """
        Return dataset length (number of samples).
        """
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """
        Inverse transform to values before scaling.
        """
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    """
    PyTorch Dataset class for ETT(minute) dataset.
    - Handles data loading, scaling, time series feature generation,
      train/validation/test split, sample extraction via getitem, etc.
    - Flexible sequence length specification with seq_len, label_len, pred_len.
    - Supports features='S' (univariate) / 'M' (multivariate) / 'MS' (multivariate+target).
    - Returns model input/output/time features via __getitem__.
    """
    def __init__(self, root_path, flag='train', setting='offline_learn', size=None,
                 features='S', data_path='ETTm1.csv', border_type='online',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        self.flag = flag
        self.setting = setting
        self.border_type = border_type

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = get_alldata(self.data_path, self.root_path)

        self.borders, border1, border2 = get_borders(self.border_type, self.data_path, self.flag, self.setting, self.seq_len, self.pred_len, len(df_raw))
        border1s, border2s = self.borders

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Get data according to flag
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, self.timeenc, freq=self.freq)
        self.data_stamp = data_stamp.astype(np.float32)

        self.data_x = torch.from_numpy(self.data_x).float()
        self.data_y = torch.from_numpy(self.data_y).float()
        self.data_stamp = torch.from_numpy(self.data_stamp).float()

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_end:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    """
    PyTorch Dataset class for arbitrary custom time series datasets.
    - Loads diverse format data using get_alldata function.
    - Flexible train/validation/test split ratio specification with ratio.
    - Supports features='S' (univariate) / 'M' (multivariate) / 'MS' (multivariate+target).
    - Returns model input/output/time features via __getitem__.
    """
    def __init__(self, root_path, flag='train', setting='offline_learn', size=None,
                 features='S', data_path='ETTh1.csv', border_type='online',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        self.flag = flag
        self.setting = setting
        self.border_type = border_type

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = get_alldata(self.data_path, self.root_path)

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        if self.features == 'S':
            cols.remove(self.target)
        cols.remove('date')

        self.borders, border1, border2 = get_borders(self.border_type, self.data_path, self.flag, self.setting, self.seq_len, self.pred_len, len(df_raw))
        border1s, border2s = self.borders

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        data = data.astype(np.float32)

        # Get data according to flag
        self.data_x = data[border1:border2]
        if self.features == 'MS':
            self.data_y = data[:, -1][border1:border2]
        else:
            self.data_y = data[border1:border2]

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, self.timeenc, freq=self.freq)
        self.data_stamp = data_stamp.astype(np.float32)

        self.data_x = torch.from_numpy(self.data_x).float()
        self.data_y = torch.from_numpy(self.data_y).float()
        self.data_stamp = torch.from_numpy(self.data_stamp).float()

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_end:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_hour_CI(Dataset_ETT_hour):
    """
    CI (Channel-Individual) extension class that extracts sequences for each feature (variable) of ETT(hour) dataset.
    - Returns sequences from normal Dataset_ETT_hour split by feature.
    - Returns sequences of only specified feature via __getitem__.
    - __len__ becomes multiple of number of features.
    """
    def __init__(self, root_path, *args, **kwargs):
        super().__init__(root_path, *args, **kwargs)
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = super().__len__()

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[s_end:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, index

    def __len__(self):
        return super().__len__() * self.enc_in


class Dataset_ETT_minute_CI(Dataset_ETT_minute):
    """
    CI (Channel-Individual) extension class that extracts sequences for each feature (variable) of ETT(minute) dataset.
    - Returns sequences from normal Dataset_ETT_minute split by feature.
    - Returns sequences of only specified feature via __getitem__.
    - __len__ becomes multiple of number of features.
    """
    def __init__(self, root_path, *args, **kwargs):
        super().__init__(root_path, *args, **kwargs)
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = super().__len__()

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[s_end:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, index

    def __len__(self):
        return super().__len__() * self.enc_in


class Dataset_Custom_CI(Dataset_Custom):
    """
    CI (Channel-Individual) extension class that extracts sequences for each feature (variable) of custom dataset.
    - Returns sequences from normal Dataset_Custom split by feature.
    - Returns sequences of only specified feature via __getitem__.
    - __len__ becomes multiple of number of features.
    """
    def __init__(self, root_path, *args, **kwargs):
        super().__init__(root_path, *args, **kwargs)
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = super().__len__()

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[s_end:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, index

    def __len__(self):
        return super().__len__() * self.enc_in


class Dataset_Recent(Dataset):
    """
    Wrapper dataset class for utilizing recent sequence information.
    - Returns data at certain index and recent data for recent_num times simultaneously.
    - Can specify interval between sequences with gap.
    - Can also add periodic fluctuations to time series data by specifying strength.
    - Returns tuple of (current data, recent data) or (recent data, current data) via __getitem__.
    """
    def __init__(self, dataset, gap: Union[int, tuple, list], **kwargs):
        super().__init__()
        self.dataset = dataset
        self.gap = gap

    def __getitem__(self, index):
        """
        Returns data at specified index (recent_batch) and its recent sequence data (current_batch).
        - Returns tuple of (data at index, data at index+gap).
        - In online learning, trains with recent_batch and validates with current_batch.
        """
        return self.dataset[index], self.dataset[index + self.gap]

    def __len__(self):
        return len(self.dataset) - self.gap

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
    指定されたファイル名とルートパスから時系列データを読み込み、
    pandas.DataFrameとして返す関数。
    - CSV, HDF5, NPZ, テキストなど複数のフォーマットに対応。
    - windデータやnycデータなど、ファイル名に応じて日付カラムの生成や整形も行う。
    - 最終的に'date'カラムが先頭になるように整形。
    """
    path = os.path.join(root_path, filename)
    if filename.endswith('.csv'):
        df = pd.read_csv(path)
        if filename.startswith('wind'):
            df['date'] = pd.date_range(start='2000-01-01', periods=len(df), freq='H')
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


class Dataset_ETT_hour(Dataset):
    """
    ETT(hour)データセット用のPyTorch Datasetクラス。
    - データの読み込み、スケーリング、時系列特徴量の生成、
      学習/検証/テスト分割、getitemによるサンプル抽出などを担当。
    - seq_len, label_len, pred_lenで系列長を柔軟に指定可能。
    - features='S'（単変量）/ 'M'（多変量）/ 'MS'（多変量+target）に対応。
    - __getitem__でモデル入力・出力・時刻特徴量を返す。
    """
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv', ratio=None, borders=None,
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False, border=None):
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
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.border = border
        self.borders = borders

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        # print("seq_len:",self.seq_len,"label_len:",self.label_len,"pred_len:",self.pred_len)

    def __read_data__(self):
        """
        データの読み込み・スケーリング・分割・時刻特徴量生成を行う内部関数。
        - train/val/testの境界はbordersまたはデフォルト値で決定。
        - featuresの種類に応じてデータ抽出。
        - スケーリングは訓練データのみでfit。
        - 時刻特徴量はutil.timefeatures.time_featuresで生成。
        """
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        if self.borders is None:
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        else:
            border1s, border2s = self.borders
        self.borders = (border1s, border2s)
        if self.border is None:
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
        else:
            border1, border2 = self.border

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

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, self.timeenc, freq=self.freq)

        self.data = data[:border2s[-1]].astype(np.float32)
        self.border = (border1, border2)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp.astype(np.float32)
        self.data_x = torch.from_numpy(self.data_x).float()
        self.data_y = torch.from_numpy(self.data_y).float()
        self.data_stamp = torch.from_numpy(self.data_stamp).float()

    def __getitem__(self, index):
        """
        指定インデックスから系列データ（入力・出力・時刻特徴量）を抽出して返す。
        - seq_x: 入力系列
        - seq_y: 予測対象系列
        - seq_x_mark: 入力系列の時刻特徴量
        - seq_y_mark: 予測対象系列の時刻特徴量
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
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        """
        データセットの長さ（サンプル数）を返す。
        """
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """
        スケーリング前の値に逆変換する。
        """
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    """
    ETT(minute)データセット用のPyTorch Datasetクラス。
    - データの読み込み、スケーリング、時系列特徴量の生成、
      学習/検証/テスト分割、getitemによるサンプル抽出などを担当。
    - seq_len, label_len, pred_lenで系列長を柔軟に指定可能。
    - features='S'（単変量）/ 'M'（多変量）/ 'MS'（多変量+target）に対応。
    - __getitem__でモデル入力・出力・時刻特徴量を返す。
    """
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv', ratio=None, borders=None,
                 target='OT', scale=True, timeenc=0, freq='t', train_only=False, border=None):
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
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.border = border
        self.borders = borders

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        if self.borders is None:
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            border1s, border2s = self.borders
        self.borders = (border1s, border2s)
        if self.border is None:
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
        else:
            border1, border2 = self.border

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

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, self.timeenc, freq=self.freq)

        self.data = data[:border2s[-1]].astype(np.float32)
        self.border = (border1, border2)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
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

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    """
    任意のカスタム時系列データセット用のPyTorch Datasetクラス。
    - get_alldata関数を用いて多様なフォーマットのデータを読み込み。
    - split_ratioで学習/検証/テストの分割比率を柔軟に指定可能。
    - features='S'（単変量）/ 'M'（多変量）/ 'MS'（多変量+target）に対応。
    - __getitem__でモデル入力・出力・時刻特徴量を返す。
    """
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv', ratio=None, borders=None,
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False, border=None):
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
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only

        self.border = border
        self.split_ratio = (0.7, 0.2) if ratio is None else ratio
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
        # print(cols)
        num_train = int(len(df_raw) * (self.split_ratio[0] if not self.train_only else 1))
        num_test = int(len(df_raw) * self.split_ratio[1])
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        self.borders = (border1s, border2s)
        if self.border is not None:
            border1, border2 = self.border
        else:
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

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
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, self.timeenc, freq=self.freq)

        data = data.astype(np.float32)
        self.data = data
        self.border = (border1, border2)

        self.data_x = data[border1:border2]
        if self.features == 'MS':
            self.data_y = data[:, -1][border1:border2]
        else:
            self.data_y = data[border1:border2]
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

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_hour_CI(Dataset_ETT_hour):
    """
    ETT(hour)データセットの各特徴量（変数）ごとに系列を抽出するCI（Channel-Individual）拡張クラス。
    - 通常のDataset_ETT_hourの系列抽出を特徴量ごとに分割して返す。
    - __getitem__で指定特徴量のみの系列を返す。
    - __len__は特徴量数倍となる。
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

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return super().__len__() * self.enc_in


class Dataset_ETT_minute_CI(Dataset_ETT_minute):
    """
    ETT(minute)データセットの各特徴量（変数）ごとに系列を抽出するCI（Channel-Individual）拡張クラス。
    - 通常のDataset_ETT_minuteの系列抽出を特徴量ごとに分割して返す。
    - __getitem__で指定特徴量のみの系列を返す。
    - __len__は特徴量数倍となる。
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

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return super().__len__() * self.enc_in


class Dataset_Custom_CI(Dataset_Custom):
    """
    カスタムデータセットの各特徴量（変数）ごとに系列を抽出するCI（Channel-Individual）拡張クラス。
    - 通常のDataset_Customの系列抽出を特徴量ごとに分割して返す。
    - __getitem__で指定特徴量のみの系列を返す。
    - __len__は特徴量数倍となる。
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

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return super().__len__() * self.enc_in


class Dataset_Recent(Dataset):
    """
    直近系列情報を活用するためのラッパーデータセットクラス。
    - あるインデックスのデータと、その直近recent_num個分のデータを同時に返す。
    - gapで系列間の間隔を指定可能。
    - strengthを指定すると時系列データに周期的な変動を加えることも可能。
    - __getitem__で(現在データ, 直近データ)または(直近データ, 現在データ)のタプルを返す。
    """
    def __init__(self, dataset, gap: Union[int, tuple, list], recent_num=1, take_post=0, strength=0, **kwargs):
        super().__init__()
        self.more = gap - recent_num + 1
        self.dataset = dataset
        self.gap = gap
        self.recent_num = recent_num
        if strength:
            print("Modify time series with strength =", strength)
            for i in range(3, len(self.dataset.data_y)):
                self.dataset.data_x[i] *= 1 + 0.1 * (i // 24 % strength)

    def _stack(self, data):
        if isinstance(data[0], np.ndarray):
            return np.vstack(data)
        else:
            return torch.stack(data, 0)

    def __getitem__(self, index):
        """
        指定インデックスのデータと、その直近系列データを返す。
        - recent_num=1の場合: (indexのデータ, index+gapのデータ)のタプルを返す。
        - recent_num>1の場合:
            ・current_data: index+gap+recent_num-1のデータ
            ・recent_data: indexからindex+recent_num-1までの直近系列をまとめて返す。
            ・current_dataがタプルでなければ(recent_data, current_data)のタプル。
            ・current_dataがタプルの場合は、各要素ごとに直近系列をまとめて返す。
        - 直近系列はtorch.stackやnp.vstackでまとめてテンソル化される。
        - オンライン学習では，recent_dataで学習し，current_dataで検証する．
        """
        if self.recent_num == 1:
            # recent_num=1の場合：単純に現在のデータとgap分先のデータを返す
            # 
            return self.dataset[index], self.dataset[index + self.gap]
        else:
            # recent_num>1の場合：複数の直近系列をまとめて返す
            # current_data: 予測対象となるデータ（gap+recent_num-1分先のデータ）
            current_data = self.dataset[index + self.gap + self.recent_num - 1]

            if not isinstance(current_data, tuple):
                # current_dataがタプルでない場合（単一のテンソル/配列）
                # indexからindex+recent_num-1までの直近系列を取得
                recent_data = tuple(self.dataset[index + n] for n in range(self.recent_num))
                # 直近系列をまとめてテンソル化（_stackでtorch.stackまたはnp.vstack）
                recent_data = self._stack(recent_data)
                # (current_data, recent_data)のタプルを返す
                return current_data, recent_data
            else:
                # current_dataがタプルの場合（複数の要素を持つ）
                # 各要素に対応する空のリストを作成
                recent_data = tuple([] for _ in range(len(current_data)))

                # 直近系列の各時点について
                for past in range(self.recent_num):
                    # index+pastのデータを取得
                    past_data_tuple = self.dataset[index + past]
                    # 各要素（seq_x, seq_y, seq_x_mark, seq_y_markなど）を対応するリストに追加
                    for j, past_data in enumerate(past_data_tuple):
                        recent_data[j].append(past_data)

                # 各要素の直近系列をまとめてテンソル化
                recent_data = tuple(self._stack(recent_d) for recent_d in recent_data)
                # (recent_data, current_data)のタプルを返す
                return recent_data, current_data

    def __len__(self):
        return len(self.dataset) - self.more

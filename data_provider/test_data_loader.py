from typing import Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


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


# ダミーデータセット: 0,1,2,...の連番を返すだけ
class DummyDataset(Dataset):
    def __init__(self, length=20):
        self.data = np.arange(length).reshape(-1, 1)
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)
    def __len__(self):
        return len(self.data)

dummy = DummyDataset(20)
print('DummyDataset:', [dummy[i].item() for i in range(len(dummy))])

# Dataset_Recentの挙動をprintで確認
def show_dataset_recent(gap, recent_num=1, take_post=0, strength=0):
    ds_recent = Dataset_Recent(dummy, gap=gap, recent_num=recent_num, take_post=take_post, strength=strength)
    print(f'--- gap={gap}, recent_num={recent_num}, take_post={take_post} ---')
    for i in range(len(ds_recent)):
        result = ds_recent[i]
        if recent_num == 1:
            x, y = result
            print(f'i={i}: x={x.item()}, y={y.item()}')
        else:
            y, x_recent = result
            print(f'i={i}: y={y.item()}, x_recent={[v.item() for v in x_recent]}')

# 例1: gap=2, recent_num=1
show_dataset_recent(gap=2, recent_num=1)

# 例2: gap=2, recent_num=3
show_dataset_recent(gap=2, recent_num=3)

# 例3: gap=1, recent_num=2
show_dataset_recent(gap=1, recent_num=2)

# 可視化: recent_num>1の場合、直近系列の値を折れ線グラフで可視化
def plot_recent_series(gap, recent_num):
    ds_recent = Dataset_Recent(dummy, gap=gap, recent_num=recent_num)
    plt.figure(figsize=(8, 4))
    for i in range(len(ds_recent)):
        y, x_recent = ds_recent[i]
        x_vals = [v.item() for v in x_recent]
        plt.plot(range(recent_num), x_vals, marker='o', label=f'i={i}, y={y.item()}')
    plt.xlabel('recent index')
    plt.ylabel('value')
    plt.title(f'Dataset_Recent: gap={gap}, recent_num={recent_num}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# 例: gap=2, recent_num=4
plot_recent_series(gap=2, recent_num=4)
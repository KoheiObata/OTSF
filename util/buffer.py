# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from typing import Tuple
from sklearn.cluster import KMeans


def reservoir_sampling(num_seen_examples: int, buffer_size: int) -> int:
    """
    Function implementing reservoir sampling algorithm.
    Used to randomly select a fixed number of samples from data stream.
    For example, when all data cannot be stored, ensures all data remains in buffer with equal probability.

    Args:
        num_seen_examples (int): Number of samples seen (processed) so far.
        buffer_size (int): Maximum size of buffer.
    Returns:
        int: Index to store current sample in buffer (returns -1 if not storing).
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def fifo(num_seen_examples: int, buffer_size: int) -> int:
    """
    FIFO (First-In-First-Out) reservoir sampling.
    Store sequentially until buffer is full, then overwrite oldest data (at front) after full.

    Args:
        num_seen_examples (int): Number of samples seen (processed) so far.
        buffer_size (int): Maximum size of buffer.
    Returns:
        int: Index to store current sample.
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples
    else:
        # Overwrite oldest data (at front)
        return num_seen_examples % buffer_size


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


def kmeans_sampling(num_seen_examples: int, buffer_size: int, features, buffer_features) -> int:
    """
    K-means based sampling. Buffer is filled sequentially with samples closest to K-means centroids.
    features: Features of new sample (1D np.array or tensor)
    buffer_features: Features currently stored in buffer (2D np.array or tensor)
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples
    # K-means clustering when buffer is full
    kmeans = KMeans(n_clusters=buffer_size, n_init=1, max_iter=10, random_state=0)
    all_features = np.concatenate([buffer_features, features[None]], axis=0)
    kmeans.fit(all_features)
    # Which cluster center is new sample closest to
    dists = np.linalg.norm(kmeans.cluster_centers_ - features, axis=1)
    closest = np.argmin(dists)
    # Index of buffer sample closest to that cluster center
    buffer_dists = np.linalg.norm(buffer_features - kmeans.cluster_centers_[closest], axis=1)
    replace_idx = np.argmax(buffer_dists)
    return replace_idx


class Buffer:
    """
    Memory buffer class for rehearsal method (past data reuse technique used in continual learning, etc.).
    Stores data (e.g., input, labels, features, etc.) in a buffer of specified size,
    and manages data using Reservoir sampling or Ring buffer method.

    Main features:
    - Add data (add_data)
    - Random sampling from buffer (get_data)
    - Buffer initialization (init_tensors)
    - Get all buffer data (get_all_data)
    - Check if buffer is empty (is_empty)
    - Clear buffer (empty)
    - Get number of data stored in buffer (len)

    Args:
        buffer_size (int): Maximum size of buffer.
        device: Device to store data (e.g., 'cpu' or 'cuda').
        n_tasks (int, optional): Number of tasks. Used in Ring buffer method.
        mode (str, optional): 'reservoir' or 'ring'. Specifies data management method.
        attr_num (int, optional): Number of attributes to store in buffer (e.g., input, labels, features, etc.).
    """
    def __init__(self, buffer_size, device, n_tasks=1, sample_selection_strategy='reservoir_sampling', attr_num=3):
        assert sample_selection_strategy in ['ring', 'reservoir_sampling', 'fifo', 'kmeans']
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.sample_selection_strategy = sample_selection_strategy

        if sample_selection_strategy == 'reservoir_sampling':
            self.functional_index = reservoir_sampling
        elif sample_selection_strategy == 'fifo':
            self.functional_index = fifo
        if sample_selection_strategy == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        if sample_selection_strategy == 'kmeans':
            self.functional_index = None  # kmeans is processed directly in add_data

        self.attr_num = attr_num
        self.buffer = []

    def init_tensors(self, *batch) -> None:
        """
        Initializes just the required tensors.
        """
        for attr in batch:
            self.buffer.append(torch.zeros((self.buffer_size, *attr.shape[1:]), dtype=torch.float32, device=self.device))

    def add_data(self, *batch):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        if self.num_seen_examples == 0:
            self.init_tensors(*batch)

        for i in range(batch[0].shape[0]):
            if self.sample_selection_strategy == 'kmeans':
                features = batch[0][i].detach().cpu().numpy()
                if self.num_seen_examples < self.buffer_size:
                    index = self.num_seen_examples
                else:
                    buffer_features = self.buffer[0][:self.buffer_size].detach().cpu().numpy()
                    index = kmeans_sampling(self.num_seen_examples, self.buffer_size, features, buffer_features)
                self.num_seen_examples += 1
                if index >= 0:
                    for j, attr in enumerate(batch):
                        self.buffer[j][index] = attr[i].detach().to(self.device)
            elif self.functional_index is not None:
                index = self.functional_index(self.num_seen_examples, self.buffer_size)
                self.num_seen_examples += 1
                if index >= 0:
                    for j, attr in enumerate(batch):
                        self.buffer[j][index] = attr[i].detach().to(self.device)

    def get_data(self, size: int) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.buffer[0].shape[0]):
            size = min(self.num_seen_examples, self.buffer[0].shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.buffer[0].shape[0]),
                                  size=size, replace=False)
        rets = []
        for attr in self.buffer:
            rets += [attr[choice]]
        return rets

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        return tuple(self.buffer)

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        self.buffer = []
        self.num_seen_examples = 0

    def len(self) ->int:
        return self.num_seen_examples

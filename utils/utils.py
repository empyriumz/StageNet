from __future__ import absolute_import
from __future__ import print_function

import threading
import numpy as np
import random

def pad_zeros(arr, min_length=None):
    """
    `arr` is an array of `np.array`s

    The function appends zeros to every `np.array` in `arr`
    to equalize their first axis lenghts.
    """
    dtype = arr[0].dtype
    max_len = max([x.shape[0] for x in arr])
    ret = [
        np.concatenate(
            [x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0
        )
        for x in arr
    ]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [
            np.concatenate(
                [x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)],
                axis=0,
            )
            for x in ret
        ]
    return np.array(ret)

class BatchDataGenerator(object):
    def __init__(
        self,
        dataloader,
        encoder,
        normalizer,
        batch_size,
        shuffle,
        return_names=False,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_names = return_names

        self._load_per_patient_data(dataloader, encoder, normalizer)

        self.steps = (len(self.data[1]) + batch_size - 1) // batch_size
        self.lock = threading.Lock()
        self.generator = self._generator()

    def _load_per_patient_data(self, dataloader, encoder, normalizer):

        N = len(dataloader._data["X"])
        Xs = []
        ys = []
        names = []
        masks = []

        for i in range(N):
            X = dataloader._data["X"][i]
            y = dataloader._data["y"][i]
            name = dataloader._data["name"][i]
            mask = dataloader._data["mask"][i]
            # one-hot encode categorical variables and imputate missing data
            X = encoder.transform(X)
            if normalizer is not None:
                X = normalizer.transform(X)

            Xs.append(X)
            ys.append(np.array(y))
            names.append(name)
            masks.append(mask)

        self.data = [[Xs, masks], ys]
        self.names = names

    def _generator(self):
        B = self.batch_size
        while True:
            if self.shuffle:
                # stupid shuffle
                N = len(self.data[1])
                order = list(range(N))
                random.shuffle(order)
                tmp_data = [[[None] * N, [None] * N], [None] * N]
                tmp_names = [None] * N
                for i in range(N):
                    tmp_data[0][0][i] = self.data[0][0][order[i]]
                    tmp_data[0][1][i] = self.data[0][1][order[i]]
                    tmp_data[1][i] = self.data[1][order[i]]
                    tmp_names[i] = self.names[order[i]]
                self.data = tmp_data
                self.names = tmp_names
            else:
                # sort entirely
                Xs = self.data[0][0]
                masks = self.data[0][1]
                ys = self.data[1]
                self.data = [[Xs, masks], ys]

            for i in range(0, len(self.data[1]), B):
                X = self.data[0][0][i : i + B]
                mask = self.data[0][1][i : i + B]
                y = self.data[1][i : i + B]
                names = self.names[i : i + B]
               
                X = pad_zeros(X)  # (B, T, D)
                mask = pad_zeros(mask)  # (B, T)
                y = pad_zeros(y)
                y = np.expand_dims(y, axis=-1)  # (B, T, 1)
                mask = np.expand_dims(mask, axis=-1)
                batch_data = ([X, mask], y)
                if not self.return_names:
                    yield batch_data
                else:
                    yield {"data": batch_data, "names": names}

    def __iter__(self):
        return self.generator

    def next(self):
        with self.lock:
            return next(self.generator)

    def __next__(self):
        return self.next()
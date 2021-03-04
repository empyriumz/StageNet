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
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle

        self._load_per_patient_data(dataloader, encoder, normalizer)
        self.steps = (len(self.data[1]) + batch_size - 1) // batch_size
        self.lock = threading.Lock()
        self.generator = self._generator()

    def _load_per_patient_data(self, dataloader, encoder, normalizer):

        N = len(dataloader._data["X"])
        Xs = []
        ys = []
      
        for i in range(N):
            X = dataloader._data["X"][i]
            y = dataloader._data["y"][i]

            # one-hot encode categorical variables and imputate missing data
            X = encoder.transform(X)
            
            # calculate intervals between measurements for each feature
            intervals = np.zeros((X.shape[0], 17))
            tmp = np.zeros(17)
            for i in range(X.shape[1]):
                # go over time direction
                # cur_ind represents the mask part in the data
                cur_ind = X[i, -17:]
                # identify the empty data spot and accumulate
                tmp += (cur_ind == 0)
                # keeps track of the the interval from last non-zero data
                intervals[i, :] = cur_ind * tmp
                # so those with non-zero data at the moment, set the timer to zero
                tmp[cur_ind == 1] = 0
            
            X = np.hstack([X, intervals.astype(np.float32)])
            if normalizer is not None:
                X = normalizer.transform(X)

            Xs.append(X)
            ys.append(y)
        
        self.data = [Xs, ys]
    
    def _generator(self):
        B = self.batch_size
        while True:
            if self.shuffle:
                # stupid shuffle
                N = len(self.data[1])
                order = list(range(N))
                random.shuffle(order)
                tmp_data = [[None] * N, [None] * N]
                for i in range(N):
                    tmp_data[0][i] = self.data[0][order[i]]
                    tmp_data[1][i] = self.data[1][order[i]]
                self.data = tmp_data
            else:
                Xs = self.data[0]
                ys = self.data[1]
                self.data = [Xs, ys]

            for i in range(0, len(self.data[1]), B):
                X = self.data[0][i : i + B]
                y = self.data[1][i : i + B]
     
                X = pad_zeros(X)  # (B, T, D)
                #y = pad_zeros(y)
                #y = np.expand_dims(y, axis=-1)  # (B, T, 1)
                batch_data = (X, y)
                
                yield {"data": batch_data}

    def __iter__(self):
        return self.generator

    def next(self):
        with self.lock:
            return next(self.generator)

    def __next__(self):
        return self.next()
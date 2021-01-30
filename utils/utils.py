from __future__ import absolute_import
from __future__ import print_function

from . import common_utils
import threading
import os
import numpy as np
import random


def preprocess_chunk(data, ts, discretizer, normalizer=None):
    data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
    if normalizer is not None:
        data = [normalizer.transform(X) for X in data]
    return data

class BatchGenerator(object):
    def __init__(
        self,
        dataloader,
        batch_size,
        shuffle = False
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.load_patient_data(dataloader)

        self.steps = (len(self.data[1]) + batch_size - 1) // batch_size
        self.lock = threading.Lock()
        self.generator = self._generator()

    def load_patient_data(self, dataloader):
        X, y = dataloader._data["X"], dataloader._data["y"]
        time_interval, names = dataloader._data['interval'], dataloader._data['name']
        mask = dataloader._data['mask']
        self.data = [X, y]
        self.names = names
        self.interval = time_interval
        self.mask = mask
    
    def _generator(self):
        B = self.batch_size
        while True:
            if self.shuffle:
                # stupid shuffle
                N = len(self.data[1])
                order = list(range(N))
                random.shuffle(order)
                tmp_data = [[None] * N, [None] * N]
                tmp_names = [None] * N
                tmp_interval = [None] * N
                tmp_mask = [None] * N
                for i in range(N):
                    tmp_data[0][i] = self.data[0][order[i]]
                    tmp_data[0][i] = self.data[0][order[i]]
                    tmp_data[1][i] = self.data[1][order[i]]
                    tmp_names[i] = self.names[order[i]]
                    tmp_interval[i] = self.interval[order[i]]
                    tmp_mask[i] = self.mask[order[i]]
                self.data = tmp_data
                self.names = tmp_names
                self.interval = tmp_interval
                self.mask = tmp_mask

            for i in range(0, len(self.data[1]), B):
                X = self.data[0][i : i + B]
                y = self.data[1][i : i + B]
                names = self.names[i : i + B]
                interval = self.interval[i : i + B]
                mask = self.mask[i : i + B]
                X = common_utils.pad_zeros(X)  # (B, T, D)
                y = common_utils.pad_zeros(y)
                mask = common_utils.pad_zeros(mask)
                interval = common_utils.pad_zeros(interval)
                X = np.expand_dims(X, axis=-1)  # (B, T, 1)
                y = np.expand_dims(y, axis=-1)  # (B, T, 1)         
                interval = np.expand_dims(interval, axis=-1)  # (B, T, 1)
                mask = np.expand_dims(mask, axis=-1)  # (B, T, 1)
                batch_data = (X, y)
                yield {"data": batch_data, "mask": mask, "names": names, "interval": interval}

    def __iter__(self):
        return self.generator

    def next(self):
        with self.lock:
            return next(self.generator)

    def __next__(self):
        return self.next()
class BatchGenDeepSupervision(object):
    def __init__(
        self,
        dataloader,
        discretizer,
        normalizer,
        batch_size,
        shuffle,
        return_names=False,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_names = return_names

        self._load_per_patient_data(dataloader, discretizer, normalizer)

        self.steps = (len(self.data[1]) + batch_size - 1) // batch_size
        self.lock = threading.Lock()
        self.generator = self._generator()

    def _load_per_patient_data(self, dataloader, discretizer, normalizer):
        timestep = discretizer._timestep

        def get_bin(t):
            eps = 1e-6
            return int(t / timestep - eps)

        N = len(dataloader._data["X"])
        Xs = []
        ts = []
        masks = []
        y = []
        names = []

        for i in range(N):
            X = dataloader._data["X"][i]
            current_t = dataloader._data["time"][i]
            current_y = dataloader._data["y"][i]
            name = dataloader._data["name"][i]

            current_y = [int(y) for y in current_y]

            T = max(current_t)
            nsteps = get_bin(T) + 1
            # mask: label time slides with non-zero data?
            mask = [0] * nsteps
            y = [0] * nsteps

            for pos, z in zip(current_t, current_y):
                mask[get_bin(pos)] = 1
                y[get_bin(pos)] = z
            # missing data is imputated?!
            # what's the point of nsteps here and the bins within the discretizer.transform()?
            X = discretizer.transform(X, end=T)
            if normalizer is not None:
                X = normalizer.transform(X)

            Xs.append(X)
            masks.append(np.array(mask))
            y.append(np.array(y))
            names.append(name)
            ts.append(current_t)

            assert np.sum(mask) > 0
            assert len(X) == len(mask) and len(X) == len(y)

        self.data = [[Xs, masks], y]
        self.names = names
        self.ts = ts

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
                tmp_interval = [None] * N
                for i in range(N):
                    tmp_data[0][0][i] = self.data[0][0][order[i]]
                    tmp_data[0][1][i] = self.data[0][1][order[i]]
                    tmp_data[1][i] = self.data[1][order[i]]
                    tmp_names[i] = self.names[order[i]]
                    tmp_interval[i] = self.ts[order[i]]
                self.data = tmp_data
                self.names = tmp_names
                self.ts = tmp_interval
            else:
                # sort entirely
                Xs = self.data[0][0]
                masks = self.data[0][1]
                y = self.data[1]
                (Xs, masks, y, self.names, self.ts) = common_utils.sort_and_shuffle(
                    [Xs, masks, y, self.names, self.ts], B
                )
                self.data = [[Xs, masks], y]

            for i in range(0, len(self.data[1]), B):
                X = self.data[0][0][i : i + B]
                mask = self.data[0][1][i : i + B]
                y = self.data[1][i : i + B]
                names = self.names[i : i + B]
                ts = self.ts[i : i + B]

                X = common_utils.pad_zeros(X)  # (B, T, D)
                mask = common_utils.pad_zeros(mask)  # (B, T)
                y = common_utils.pad_zeros(y)
                y = np.expand_dims(y, axis=-1)  # (B, T, 1)
                batch_data = ([X, mask], y)
                if not self.return_names:
                    yield batch_data
                else:
                    yield {"data": batch_data, "names": names, "ts": ts}

    def __iter__(self):
        return self.generator

    def next(self):
        with self.lock:
            return next(self.generator)

    def __next__(self):
        return self.next()


def save_results(names, ts, pred, y_true, path):
    common_utils.create_directory(os.path.dirname(path))
    with open(path, "w") as f:
        f.write("stay,period_length,prediction,y_true\n")
        for (name, t, x, y) in zip(names, ts, pred, y_true):
            f.write("{},{:.6f},{:.6f},{}\n".format(name, t, x, y))

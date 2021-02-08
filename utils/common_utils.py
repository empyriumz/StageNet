from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
import random
import pandas as pd
import threading

class DeepSupervisionDataLoader:
    r"""
    Data loader for decompensation and length of stay task.
    Reads all the data for one patient at once.

    Parameters
    ----------
    dataset_dir : str
        Directory where timeseries files are stored.
    listfile : str
        Path to a listfile. If this parameter is left `None` then
        `dataset_dir/listfile.csv` will be used.
    """

    def __init__(self, dataset_dir, listfile=None, small_part=False):

        self._dataset_dir = dataset_dir
        if listfile is None:
            listfile_path = os.path.join(dataset_dir, "listfile.csv")
        else:
            listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()[1:]  # skip the header

        self._data = [line.split(",") for line in self._data]
        self._data = [(x, float(t), y) for (x, t, y) in self._data]
        self._data = sorted(self._data)

        mas = {"X": [], "ts": [], "ys": [], "name": []}
        i = 0
        while i < len(self._data):
            j = i
            cur_stay = self._data[i][0]
            cur_ts = []
            cur_labels = []
            while j < len(self._data) and self._data[j][0] == cur_stay:
                cur_ts.append(self._data[j][1])
                cur_labels.append(self._data[j][2])
                j += 1

            cur_X = self._read_timeseries(cur_stay)
            mas["X"].append(cur_X)
            mas["ts"].append(cur_ts)
            mas["ys"].append(cur_labels)
            mas["name"].append(cur_stay)

            i = j
            if small_part and len(mas["name"]) == 256:
                break

        self._data = mas

    def _read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(",")
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(",")
                ret.append(np.array(mas))
        return np.stack(ret)

class DataLoader:
    r"""
    Data loader for decompensation task.
    Reads all the data for one patient at once.

    Parameters
    ----------
    dataset_dir : str
        Directory where timeseries files are stored.
    file_list : str
        Path to a file_list. If this parameter is left `None` then
        `dataset_dir/file_list.csv` will be used.
    """

    def __init__(self, dataset_dir, batch_size, file_list=None, shuffle=False):
        self._dataset_dir = dataset_dir
        if file_list is None:
            file_list_path = os.path.join(self._dataset_dir, "file_list.csv")
        else:
            file_list_path = file_list
        df = pd.read_csv(file_list_path)
        df = df.sort_values(by=['stay', 'period_length'])
        group = df.groupby('stay').agg(list)
        
        data = {"X": [],  "time": [], "mask": [], "interval": [], "y": [], "name": []}
        name = list(group.index)
        data['name'] = name
        # X and y dimension don't match due to the special data structure
        # we need to manually adjust the data to make the two match
        # make the assumption that y doesn't change
        y = group['y_true'].apply(lambda x: x[-1]).to_list()
        y_true = dict(zip(name, y))
        for file_name in data['name']:
            tmp_df = pd.read_csv(self._dataset_dir+"/"+file_name)
            current_data = tmp_df[['Hours','Diastolic blood pressure']].dropna()
            if len(current_data) == 0:
                continue
            current_data = current_data.sort_values(by=['Hours'])
            current_data['interval'] = current_data['Hours'].diff().fillna(0)
            data["time"].append(current_data['Hours'].values)
            data["X"].append(current_data['Diastolic blood pressure'].values)
            tmp = current_data['interval'].values
            data["interval"].append(tmp)
            data["mask"].append(np.ones_like(tmp))
            data["y"].append(y_true[file_name] * np.ones_like(tmp))
           
        self.interval, self.names = data['interval'], data['name']
        self.mask = data['mask']
        self.data = [data["X"], data["y"]]
        
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.steps = (len(self.data[1]) + batch_size - 1) // batch_size
        self.lock = threading.Lock()
        self.generator = self._generator()

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
                X = pad_zeros(X)  # (B, T, D)
                y = pad_zeros(y)
                mask = pad_zeros(mask)
                interval = pad_zeros(interval)
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

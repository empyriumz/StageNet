from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
import json
import random
import pandas as pd

from .feature_extractor import extract_features


def convert_to_dict(data, header, channel_info):
    """ convert data from readers output in to array of arrays format """
    ret = [[] for _ in range(data.shape[1] - 1)]
    for i in range(1, data.shape[1]):
        ret[i - 1] = [(t, x) for (t, x) in zip(data[:, 0], data[:, i]) if x != ""]
        channel = header[i]
        if len(channel_info[channel]["possible_values"]) != 0:
            ret[i - 1] = list(
                map(lambda x: (x[0], channel_info[channel]["values"][x[1]]), ret[i - 1])
            )
        ret[i - 1] = list(map(lambda x: (float(x[0]), float(x[1])), ret[i - 1]))
    return ret


def extract_features_from_rawdata(chunk, header, period, features):
    with open(
        os.path.join(os.path.dirname(__file__), "resources/channel_info.json")
    ) as channel_info_file:
        channel_info = json.loads(channel_info_file.read())
    data = [convert_to_dict(X, header, channel_info) for X in chunk]
    return extract_features(data, period, features)


def read_chunk(reader, chunk_size):
    data = {}
    for i in range(chunk_size):
        ret = reader.read_next()
        for k, v in ret.items():
            if k not in data:
                data[k] = []
            data[k].append(v)
    data["header"] = data["header"][0]
    return data


def sort_and_shuffle(data, batch_size):
    """Sort data by the length and then make batches and shuffle them.
    data is tuple (X1, X2, ..., Xn) all of them have the same length.
    Usually data = (X, y).
    """
    assert len(data) >= 2
    data = list(zip(*data))

    random.shuffle(data)

    old_size = len(data)
    rem = old_size % batch_size
    head = data[: old_size - rem]
    tail = data[old_size - rem :]
    data = []

    head.sort(key=(lambda x: x[0].shape[0]))

    mas = [head[i : i + batch_size] for i in range(0, len(head), batch_size)]
    random.shuffle(mas)

    for x in mas:
        data += x
    data += tail

    data = list(zip(*data))
    return data
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
        blacklist = [
        # Criterion for exclusion: more than 1000 distinct timepoints
        # In training data
        '73129_episode2_timeseries.csv', '48123_episode2_timeseries.csv',
        '76151_episode2_timeseries.csv', '41493_episode1_timeseries.csv',
        '65565_episode1_timeseries.csv', '55205_episode1_timeseries.csv',
        '41861_episode1_timeseries.csv', '58242_episode4_timeseries.csv',
        '54073_episode1_timeseries.csv', '46156_episode1_timeseries.csv',
        '55639_episode1_timeseries.csv', '89840_episode1_timeseries.csv',
        '43459_episode1_timeseries.csv', '10694_episode2_timeseries.csv',
        '51078_episode2_timeseries.csv', '90776_episode1_timeseries.csv',
        '89223_episode1_timeseries.csv', '12831_episode2_timeseries.csv',
        '80536_episode1_timeseries.csv',
        # In validation data
        '78515_episode1_timeseries.csv', '62239_episode2_timeseries.csv',
        '58723_episode1_timeseries.csv', '40187_episode1_timeseries.csv',
        '79337_episode1_timeseries.csv',
        # In testing data
        '51177_episode1_timeseries.csv', '70698_episode1_timeseries.csv',
        '48935_episode1_timeseries.csv', '54353_episode2_timeseries.csv',
        '19223_episode2_timeseries.csv', '58854_episode1_timeseries.csv',
        '80345_episode1_timeseries.csv', '48380_episode1_timeseries.csv'
        ]
        self._dataset_dir = dataset_dir
        if listfile is None:
            listfile_path = os.path.join(dataset_dir, "listfile.csv")
        else:
            listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()[1:]  # skip the header

        self._data = [line.split(",") for line in self._data]
        self._data = [(x, y) for (x, y) in self._data]
        #self._data = sorted(self._data)

        mas = {"X": [], "y": [], "name": [], "mask": []}
        for i in range(len(self._data)):
            cur_stay = self._data[i][0]
            if cur_stay in blacklist:
                continue
            cur_labels = int(self._data[i][1])
            cur_X = self._read_timeseries(cur_stay)
            mas["X"].append(cur_X)
            mask = np.ones_like(cur_X[:, 0], dtype=int)
            mas["mask"].append(mask)
            y = cur_labels * np.ones_like(cur_X[:, 0], dtype=int)
            mas["y"].append(y)
            mas["name"].append(cur_stay)
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
    listfile : str
        Path to a listfile. If this parameter is left `None` then
        `dataset_dir/listfile.csv` will be used.
    """

    def __init__(self, dataset_dir, listfile=None):
        self._dataset_dir = dataset_dir
        if listfile is None:
            listfile_path = os.path.join(dataset_dir, "listfile.csv")
        else:
            listfile_path = listfile
        df = pd.read_csv(listfile_path)
        #df = df.sort_values(by=['stay'])
        mas = {"X": [], "y": [], "name": [], "mask": []}
        mas['name'] = df['stay'].values
        mas["y"] = df['y_true'].values
        for file_name in mas['name']:
            tmp_df = pd.read_csv(self._dataset_dir+"/"+file_name)
            tmp_df = tmp_df.dropna(how="all")
            current_X = tmp_df.to_numpy()
            mask = np.ones_like(current_X[:, 0])
            mas["X"].append(current_X)
            mas["mask"].append(mask)
        self._data = mas
            
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


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

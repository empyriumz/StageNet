from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
import pandas as pd
class MortalityDataLoader:
    r"""
    Data loader for mortality prediction task.
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
        self.listfile = listfile
        with open(self.listfile, "r") as lfile:
            self._data = lfile.readlines()[1:]  # skip the header

        self._data = [line.split(",") for line in self._data]
        self._data = [(x, y) for (x, y) in self._data]
    
        mas = {"X": [], "y": []}
        for i in range(len(self._data)):
            cur_stay = self._data[i][0]
            if cur_stay in blacklist:
                continue
            y_labels = int(self._data[i][1])
            cur_X = self._read_timeseries(cur_stay)
            mas["X"].append(cur_X)
            mas["y"].append(y_labels)
            if small_part and len(mas["y"]) == 256:
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

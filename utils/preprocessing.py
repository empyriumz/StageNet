from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import platform
import pickle
import json
import os


class Discretizer:
    def __init__(
        self,
        store_masks=True,
        impute_strategy="zero",
        start_time="zero",
        config_path=os.path.join(
            os.path.dirname(__file__), "resources/discretizer_config.json"
        ),
    ):

        with open(config_path) as f:
            config = json.load(f)
            self._id_to_channel = config["id_to_channel"]
            self._channel_to_id = dict(
                zip(self._id_to_channel, range(len(self._id_to_channel)))
            )
            self._is_categorical_channel = config["is_categorical_channel"]
            self._possible_values = config["possible_values"]
            self._normal_values = config["normal_values"]

        self._header = ["Hours"] + self._id_to_channel
        self._store_masks = store_masks
        self._start_time = start_time
        self._impute_strategy = impute_strategy
        self.vector_len, self.code_pos = self._get_code_info()
        # for statistics
        self._done_count = 0
        self._empty_bins_sum = 0
        self._unused_data_sum = 0
    
    def _get_code_info(self):
        """determine how long will the clinical feature vector be and 
        the starting column index for writing each columns in the data matrix

        Returns:
            [type]: [description]
        """        
        N_channels = len(self._id_to_channel)
        vec_len = 0
        begin_pos = [0] * N_channels
        end_pos = [0] * N_channels
        for i in range(N_channels):
            channel = self._id_to_channel[i]
            begin_pos[i] = vec_len
            if self._is_categorical_channel[channel]:
                end_pos[i] = begin_pos[i] + len(self._possible_values[channel])
            else:
                end_pos[i] = begin_pos[i] + 1
            vec_len = end_pos[i]
        return vec_len, begin_pos
        
    def transform(self, X, header=None, end=None):
        if header is None:
            header = self._header
        assert header[0] == "Hours"
        eps = 1e-6

        N_channels = len(self._id_to_channel)
        ts = [float(row[0]) for row in X]
        for i in range(len(ts) - 1):
            assert ts[i] < ts[i + 1] + eps

        if self._start_time == "relative":
            first_time = ts[0]
        elif self._start_time == "zero":
            first_time = 0
        else:
            raise ValueError("start_time is invalid")

        if end is None:
            max_hours = max(ts) - first_time
        else:
            max_hours = end - first_time

        N_bins = X.shape[0]

        # record all data, both categorical and continous variables into one matrix "data"
        data = np.zeros(shape=(N_bins, self.vector_len), dtype=float)
        # mask: denote which column has non-zero data for each bin??
        mask = np.zeros(shape=(N_bins, N_channels), dtype=int)
        original_value = [["" for _ in range(N_channels)] for _ in range(N_bins)]
        total_data = 0
        unused_data = 0

        def write(data, bin_id, channel, value):
            channel_id = self._channel_to_id[channel]
            if self._is_categorical_channel[channel]:
                category_id = self._possible_values[channel].index(value)
                N_values = len(self._possible_values[channel])
                one_hot = np.zeros((N_values,))
                one_hot[category_id] = 1
                for pos in range(N_values):
                    data[bin_id, self.code_pos[channel_id] + pos] = one_hot[pos]
            else:
                data[bin_id, self.code_pos[channel_id]] = float(value)

        for i, row in enumerate(X):
            t = float(row[0]) - first_time
            if t > max_hours + eps:
                continue
            
            for j in range(1, len(row)):
                if row[j] == "":
                #if row[j] != row[j]: # find NaN values
                    continue
                channel = header[j]
                channel_id = self._channel_to_id[channel]

                total_data += 1
                if mask[i][channel_id] == 1:
                    unused_data += 1
                # record position with non-zero data
                mask[i][channel_id] = 1

                write(data, i, channel, row[j])
                original_value[i][channel_id] = row[j]

        # impute missing values
        if self._impute_strategy not in ["zero", "normal_value", "previous", "next"]:
            raise ValueError("impute strategy is invalid")

        if self._impute_strategy in ["normal_value", "previous"]:
            prev_values = [[] for _ in range(len(self._id_to_channel))]
            for bin_id in range(N_bins):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        prev_values[channel_id].append(
                            original_value[bin_id][channel_id]
                        )
                        continue
                    if self._impute_strategy == "normal_value":
                        imputed_value = self._normal_values[channel]
                    if self._impute_strategy == "previous":
                        if len(prev_values[channel_id]) == 0:
                            imputed_value = self._normal_values[channel]
                        else:
                            imputed_value = prev_values[channel_id][-1]
                    write(data, bin_id, channel, imputed_value)

        if self._impute_strategy == "next":
            prev_values = [[] for _ in range(len(self._id_to_channel))]
            for bin_id in range(N_bins - 1, -1, -1):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        prev_values[channel_id].append(
                            original_value[bin_id][channel_id]
                        )
                        continue
                    if len(prev_values[channel_id]) == 0:
                        imputed_value = self._normal_values[channel]
                    else:
                        imputed_value = prev_values[channel_id][-1]
                    write(data, bin_id, channel, imputed_value)

        empty_bins = np.sum([1 - min(1, np.sum(mask[i, :])) for i in range(N_bins)])
        self._done_count += 1
        self._empty_bins_sum += empty_bins / (N_bins + eps)
        self._unused_data_sum += unused_data / (total_data + eps)

        if self._store_masks:
            data = np.hstack([data, mask.astype(np.float32)])
        return data

    def create_header(self):
        header = []
        for channel in self._id_to_channel:
            if self._is_categorical_channel[channel]:
                values = self._possible_values[channel]
                for value in values:
                    header.append(channel + "->" + value)
            else:
                header.append(channel)

        if self._store_masks:
            for i in range(len(self._id_to_channel)):
                channel = self._id_to_channel[i]
                header.append("mask->" + channel)

        header = ",".join(header)
        return header
        
    def print_statistics(self):
        print("statistics of discretizer:")
        print("\tconverted {} examples".format(self._done_count))
        print(
            "\taverage unused data = {:.2f} percent".format(
                100.0 * self._unused_data_sum / self._done_count
            )
        )
        print(
            "\taverage empty  bins = {:.2f} percent".format(
                100.0 * self._empty_bins_sum / self._done_count
            )
        )


class Normalizer:
    def __init__(self, fields=None):
        self._means = None
        self._stds = None
        self._fields = None
        if fields is not None:
            self._fields = [col for col in fields]

        self._sum_x = None
        self._sum_sq_x = None
        self._count = 0

    def _feed_data(self, x):
        x = np.array(x)
        self._count += x.shape[0]
        if self._sum_x is None:
            self._sum_x = np.sum(x, axis=0)
            self._sum_sq_x = np.sum(x ** 2, axis=0)
        else:
            self._sum_x += np.sum(x, axis=0)
            self._sum_sq_x += np.sum(x ** 2, axis=0)

    def _save_params(self, save_file_path):
        eps = 1e-7
        with open(save_file_path, "wb") as save_file:
            N = self._count
            self._means = 1.0 / N * self._sum_x
            self._stds = np.sqrt(
                1.0
                / (N - 1)
                * (
                    self._sum_sq_x
                    - 2.0 * self._sum_x * self._means
                    + N * self._means ** 2
                )
            )
            self._stds[self._stds < eps] = eps
            pickle.dump(
                obj={"means": self._means, "stds": self._stds},
                file=save_file,
                protocol=2,
            )

    def load_params(self, load_file_path):
        with open(load_file_path, "rb") as load_file:
            if platform.python_version()[0] == "2":
                dct = pickle.load(load_file)
            else:
                dct = pickle.load(load_file, encoding="latin1")
            self._means = dct["means"]
            self._stds = dct["stds"]

    def transform(self, X):
        if self._fields is None:
            fields = range(X.shape[1])
        else:
            fields = self._fields
        ret = 1.0 * X
        for col in fields:
            ret[:, col] = (X[:, col] - self._means[col]) / self._stds[col]
        return ret

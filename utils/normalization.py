import numpy as np
import os
import pickle

from common_utils import MortalityDataLoader
from preprocessing import OneHotEncoder
from utils import BatchDataGenerator

data_loader = MortalityDataLoader(
        dataset_dir=os.path.join("../mortality_data", "train"),
        listfile=os.path.join("../mortality_data", "all_list.csv"),
    )

encoder = OneHotEncoder(
        store_masks=False,
        impute_strategy="previous",
        start_time="zero",
    )

data_gen = BatchDataGenerator(
        data_loader,
        encoder,
        None,
        256,
        shuffle=True,
        return_names=True,
    )

mean = []
std = []
for k in range(data_gen.data[0][0][0].shape[1]): # total feature dimensions after one-hot encoding
    j = []
    for i in data_gen.data[0][0]:
        j.append(i[:, k])
    out = np.concatenate(j).ravel()
    mean.append(out.mean())
    std.append(out.std())

data = {"mean": mean, "std": std}


with open('./resources/mortality_normalizer.pkl', 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
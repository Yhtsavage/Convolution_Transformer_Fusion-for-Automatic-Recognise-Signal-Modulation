import os
import gc
import itertools
import pickle
import torch
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, Dict
from torch.utils.data import DataLoader, random_split

# Modulation types
MODULATIONS = {
    "QPSK": 0,
    "8PSK": 1,
    "AM-DSB": 2,
    "AM-SSB": 3,
    "QAM16": 4,
    "GFSK": 5,
    "QAM64": 6,
    "PAM4": 7,
    "CPFSK": 8,
    "BPSK": 9,
    "WBFM": 10,
}

# Signal-to-Noise Ratios
SNRS = [
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18,
    -20, -18, -16, -14, -12, -10, -8, -6, -4, -2
]


class RadioML2016(torch.utils.data.Dataset):
    #URL = "http://opendata.deepsig.io/datasets/2016.10/RML2016.10b.tar.bz2"
    modulations = MODULATIONS
    snrs = SNRS

    def __init__(
            self,
            data_dir: str = "./RML2016.10a",
            file_name: str = "RML2016.10a_dict.pkl"
    ):
        self.file_name = file_name
        self.data_dir = data_dir
        self.n_classes = len(self.modulations)
        self.X, self.y = self.load_data()
        gc.collect()

    def load_data(self):
        """ Load data from file """
        print("Loading dataset from file...")
        with open(os.path.join(self.data_dir, self.file_name), "rb") as f:
            data = pickle.load(f, encoding="latin1")

        X, y = [], []
        print("Processing dataset")
        for mod, snr in tqdm(list(itertools.product(self.modulations, self.snrs))):
            X.append(data[(mod, snr)])

            for i in range(data[(mod, snr)].shape[0]):
                y.append((mod, snr))

        X = np.vstack(X)
        return X, y

    def __getitem__(self, idx):
        """ Load a batch of input and labels """
        x, (mod, snr) = self.X[idx], self.y[idx]
        y = self.modulations[mod]
        x, y = torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
        x = x.to(torch.float).unsqueeze(0)
        return x, y

    def __len__(self):
        return self.X.shape[0]

    def get_signals(self, mod, snr):
        """ Return signals of a certain modulation or signal-to-noise ratio """

        # If None then it means all mods or snrs
        if mod is None:
            modulations = self.modulations.copy()
        if snr is None:
            snrs = self.snrs.copy()

        # If single mod or snr then convert to list to make iterable
        if not isinstance(mod, List):
            modulations = [mod]
        if not isinstance(snr, List):
            snrs = [snr]

        # Aggregate signals into a dictionary
        X = {}
        for mod, snr in list(itertools.product(modulations, snrs)):
            X[(mod, snr)] = []
            for idx, (m, s) in enumerate(self.y):
                if m == mod and s == snr:
                    X[(mod, snr)].append(np.expand_dims(self.X[idx, ...], axis=0))

            X[(mod, snr)] = np.concatenate(X[(mod, snr)], axis=0)

        return X

SPLIT = 0.7
dataset = RadioML2016()    
total = len(dataset)
lengths = [int(len(dataset) * SPLIT)]
lengths.append(total - lengths[0])    
print("Splitting into {} train and {} val".format(lengths[0], lengths[1]))
train_set, val_set = random_split(dataset, lengths)
lengths_test = [int((lengths[1])*(2/3))]
lengths_test.append(lengths[1]-lengths_test[0])
val_set, test_set = random_split(val_set, lengths_test)
#DataLoader_whole = DataLoader(dataset, batch_size=2048, shuffle=True)
DataLoader_train = DataLoader(train_set, batch_size=2048, shuffle=True)
DataLoader_val = DataLoader(val_set, batch_size=1024, shuffle=True)
if __name__ == "__main__":
    n_train = 200000 * 0.5
    np.random.seed(2016)
    testidx = sorted(np.random.choice(range(0,200000), size=int(n_train), replace=False))
    # testingX = torch.from_numpy(dataset.X.reshape(200000, 128, 2)).to(torch.float).unsqueeze(1) # 10000, 1, 128, 2
    #[x for x in range(200000) if dataset.y[x][1] == 0] get index
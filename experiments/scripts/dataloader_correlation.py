import torch
import h5py
import numpy as np

class Dataset(torch.utils.data.Dataset):

    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            self.fmap = np.array(f["/MOT17-02/fmap"]).astype("float32")
            self.fmap_enlarged = np.array(f["/MOT17-02/fmap_enlarged"]).astype("float32")
            self.labels = labels = np.array(f["/MOT17-02/labels"]).astype("float32")
            self.length = self.fmap.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.fmap[index], self.fmap_enlarged[index], self.labels[index]
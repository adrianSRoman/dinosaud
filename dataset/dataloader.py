import torch
from torch.utils.data import Dataset
import h5py

class Dataset(Dataset):
    def __init__(self, hdf5_path, transform=None):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.transform = transform

        # Open once to read metadata
        with h5py.File(self.hdf5_path, 'r') as f:
            self.length = f['mic'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with h5py.File(self.hdf5_path, 'r') as f:
            mic = f['mic'][index]  # (C, T)
            foa = f['foa'][index]
            filename = f['filenames'][index].decode('utf-8')

        if self.transform:
            mic = self.transform(mic)

        return mic, foa, filename

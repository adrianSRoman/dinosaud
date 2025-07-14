import os

import librosa
from torch.utils import data

from util.utils import sample_fixed_length_data_aligned


class Dataset(data.Dataset):
    def __init__(self,
                 dataset,
                 limit=None,
                 offset=0,
                 sample_length=16384,
                 mode="train"):
        """Construct dataset for training and validation.
        Args:
            dataset (str): *.txt, the path of the dataset list file. See "Notes."
            limit (int): Return at most limit files in the list. If None, all files are returned.
            offset (int): Return files starting at an offset within the list. Use negative values to offset from the end of the list.
            sample_length(int): The model only supports fixed-length input. Use sample_length to specify the feature size of the input.
            mode(str): If mode is "train", return fixed-length signals. If mode is "validation", return original-length signals.

        Notes:
            dataset list file：
            <mic_1_path><space><foa_1_path>
            <mic_2_path><space><foa_2_path>
            ...
            <mic_n_path><space><foa_n_path>

            e.g.
            /train/mic/a.wav /train/foa/a.wav
            /train/mic/b.wav /train/foa/b.wav
            ...

        Return:
            (mixture signals, foa signals, filename)
        """
        super(Dataset, self).__init__()
        dataset_list = [line.rstrip('\n') for line in open(os.path.abspath(os.path.expanduser(dataset)), "r")]

        dataset_list = dataset_list[offset:]
        if limit:
            dataset_list = dataset_list[:limit]

        assert mode in ("train", "validation"), "Mode must be one of 'train' or 'validation'."

        self.length = len(dataset_list)
        self.dataset_list = dataset_list
        self.sample_length = sample_length
        self.mode = mode

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        mic_path, foa_path = self.dataset_list[item].split(" ")
        filename = os.path.splitext(os.path.basename(mic_path))[0]
        mic_sig, _ = librosa.load(os.path.abspath(os.path.expanduser(mic_path)), sr=None, mono=False)
        foa_sig, _ = librosa.load(os.path.abspath(os.path.expanduser(foa_path)), sr=None, mono=False)

        if self.mode == "train":
            # The input of model should be fixed-length in the training.
            mic_sig, foa_sig = sample_fixed_length_data_aligned(mic_sig, foa_sig, self.sample_length)
            return mic_sig, foa_sig, filename
        else:
            return mic_sig, foa_sig, filename

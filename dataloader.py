import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset


categories = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
              'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
              'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
              'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
              'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
              'Clapping']


def ids_to_multinomial(ids):
    """ label encoding

    Returns:
      1d array, multimonial representation, e.g. [1,0,1,0,0,...]
    """
    id_to_idx = {id: index for index, id in enumerate(categories)}

    y = np.zeros(len(categories))
    for id in ids:
        index = id_to_idx[id]
        y[index] = 1
    return y


class LLP_dataset(Dataset):
    def __init__(self, label, audio_dir, video_dir, st_dir,
                 transform=None, a_smooth=1.0, v_smooth=0.9):
        self.df = pd.read_csv(label, header=0, sep='\t')
        self.filenames = self.df["filename"]
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.st_dir = st_dir
        self.transform = transform
        self.a_smooth = a_smooth
        self.v_smooth = v_smooth

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        row = self.df.loc[idx, :]
        name = row[0][:11]
        audio = np.load(os.path.join(self.audio_dir, name + '.npy'))
        video_s = np.load(os.path.join(self.video_dir, name + '.npy'))
        video_st = np.load(os.path.join(self.st_dir, name + '.npy'))
        ids = row[-1].split(',')
        label = ids_to_multinomial(ids)

        Pa = self.a_smooth * label + (1 - self.a_smooth) * 0.5
        Pv = self.v_smooth * label + (1 - self.v_smooth) * 0.5

        sample = {'audio': audio, 'video_s': video_s, 'video_st': video_st,
                  'label': label, 'Pa': Pa, 'Pv': Pv, 'idx': np.array([idx])}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor:
    def __call__(self, sample):
        tensor = dict()
        for key in sample:
            tensor[key] = torch.from_numpy(sample[key])
        return tensor

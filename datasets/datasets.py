import numpy as np
import h5py
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class Dataset(object):
    def __init__(self, loc):
        self.dataset = torch.load(loc).float().div(255).view(-1, 1, 64, 64)

    def __len__(self):
        return self.dataset.size(0)

    @property
    def ndim(self):
        return self.dataset.size(1)

    def __getitem__(self, index):
        return self.dataset[index]


class Faces(Dataset):
    LOC = 'data/basel_face_renders.pth'

    def __init__(self):
        return super(Faces, self).__init__(self.LOC)

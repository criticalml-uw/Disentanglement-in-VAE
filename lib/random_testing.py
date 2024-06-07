import torch
import numpy as np
import h5py
class d3Shapes(object):
    def __init__(self, test = True):
        loc =  '../../Datasets/real_3d_complicated/real3d_complicated_shapes_ordered.npz'
        self.test = test
        if self.test == True:
            self.data = np.load(loc, encoding='latin1')['images']
            '''self.data = torch.from_numpy(self.data)'''
        else: 
            self.data = np.load(loc, encoding='latin1')['images']
            self.data = torch.from_numpy(self.data).float().div(255)
            self.data = torch.permute(self.data, (0,3,1,2))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        im = self.data[index]
        if self.test == True:
            im = im / 255.
            im = im.astype(np.float32)
            im = torch.from_numpy(im)
            print(im.shape)
            if(len(im.shape) == 4):
                im = torch.permute(im, (0, 3, 1, 2))
            else:
                im = torch.permute(im, (2, 0, 1))
        '''print(torch.min(x), torch.max(x))'''
        return im

import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch.utils.data import DataLoader
train_set = d3Shapes(test = True)
train_loader = DataLoader(dataset=train_set,
        batch_size=32, shuffle=True)
dataset = train_loader.dataset
print(len(dataset))
print(dataset[100].shape)
print(torch.max(dataset[1000]), torch.min(dataset[1000]))
img = dataset[100]
save_image(dataset[100], "img_realimg.png")

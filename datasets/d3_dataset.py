import numpy as np
import h5py
import torch

class d3Shapes(object):
    def __init__(self):
        loc = '../Datasets/3d_shape_dataset/3dshapes.h5'
        with h5py.File(loc, "r") as f:
            print(f.keys())
            a = list(f.keys())[0]
            '''self.imgs = torch.from_numpy(np.array(f[a])).float()'''
            self.imgs = torch.from_numpy(np.array(f[a])).float().div(255).view(-1, 3, 64, 64)
            '''numpy_array = np.array(f[a])'''
            '''self.imgs = torch.from_numpy(np.array(f[a]) - np.mean(np.array(f[a]))).float().div(np.std(np.array(f[a]))).view(-1, 3, 64, 64)'''
            print(type(self.imgs))

    def __len__(self):
        return self.imgs.size(0)

    def __getitem__(self, index):
        x = self.imgs[index]
        print(torch.min(x), torch.max(x))
        return x 
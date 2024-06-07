import torch 
import numpy as np

class mpi3D(object):
    def __init__(self, test = True):
        loc =  '../Datasets/real_3d_complicated/real3d_complicated_shapes_ordered.npz'
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
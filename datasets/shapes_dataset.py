import numpy as np
import torch

class Shapes(object):

    def __init__(self, dataset_zip=None):
        loc = '../../Datasets/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        if dataset_zip is None:
            self.dataset_zip = np.load(loc, encoding='latin1')
        else:
            self.dataset_zip = dataset_zip
        self.imgs = torch.from_numpy(self.dataset_zip['imgs']).float()
        #print(self.imgs400)
        '''for i in range(len(self.imgs[400])):
            for j in range(len(self.imgs[400,i])):
                if(self.imgs[400,i,j] != torch.Tensor([0])[0]):
                    print(self.imgs[400,i,j])'''

    def __len__(self):
        return self.imgs.size(0)

    def __getitem__(self, index):
        x = self.imgs[index].view(1, 64, 64)
        '''print(torch.min(x), torch.max(x))'''
        return x
    


#shapes = Shapes()
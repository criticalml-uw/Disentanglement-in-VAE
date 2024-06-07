import numpy as np
import h5py
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class Shapes(object):

    def __init__(self, dataset_zip=None):
        loc = '../Datasets/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        if dataset_zip is None:
            self.dataset_zip = np.load(loc, encoding='latin1')
        else:
            self.dataset_zip = dataset_zip
        self.imgs = torch.from_numpy(self.dataset_zip['imgs']).float()

    def __len__(self):
        return self.imgs.size(0)

    def __getitem__(self, index):
        x = self.imgs[index].view(1, 64, 64)
        '''print(torch.min(x), torch.max(x))'''
        return x

class d3Shapes(object):
    def __init__(self, test = True):
        loc = '../Datasets/3d_shape_dataset/3dshapes.h5'
        self.test = test
        if self.test == True:
            dataset = h5py.File('../Datasets/3d_shape_dataset/3dshapes.h5', 'r')
            print(dataset.keys())
            self.images = dataset['images']
            self.labels =  dataset['labels']
        else: 
            with h5py.File(loc, "r") as f:
                print(f.keys())
                a = list(f.keys())[0]
                self.images = torch.from_numpy(np.array(f[a])).float().div(255).view(-1, 64, 64, 3)
                self.images = torch.permute(self.images, (0, 3, 1, 2))
                #self.imgs = torch.from_numpy((f[a]/255).astype(np.float32)).view(-1, 64, 64,3)
                print(self.images.shape)
                print(type(self.images))

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        im = self.images[index]
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


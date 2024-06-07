import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from .d3_dataset import d3Shapes
from .shapes_dataset import Shapes
from .datasets import Faces 
from .mpi import mpi3D



# for loading and batching datasets
def setup_data_loaders(args, use_cuda=False, test = False):
    if args.dataset == 'shapes':
        train_set = Shapes()
    elif args.dataset == 'faces':
        train_set = Faces()
    elif args.dataset == 'mnist' and test == False:
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        print(type(train_set))
    elif args.dataset == 'mnist' and test == True:
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif args.dataset == '3dshapes':
        train_set = d3Shapes(test)
    elif args.dataset == 'mpi3D':
        train_set = mpi3D(test)
    else:
        raise ValueError('Unknown dataset ' + str(args.dataset))

    '''kwargs = {'num_workers': 4, 'pin_memory': use_cuda}**kwargs'''
    train_loader = DataLoader(dataset=train_set,
        batch_size=args.batch_size, shuffle=True)
    return train_loader
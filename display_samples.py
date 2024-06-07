from metric_helpers.loader import load_model_and_dataset
import argparse
import torch
from torch.autograd import Variable
from vae_quant_cpy import setup_data_loaders
from vae_quant_cpy import VAE
from torchvision.utils import save_image
import torchvision.transforms as T
import numpy as np
import torch.nn.functional as F

import lib.dist as dist
from lib.flows import FactorialNormalizingFlow

import matplotlib.pyplot as plt


def display_samples(model, x, mnist = False, d3 = False, requirement = 'latent_walks'):
    '''global win_samples, win_test_reco, win_latent_walk'''
    if requirement == 'random_sample':
        # plot random samples
        sample_mu = model.model_sample(batch_size=100).sigmoid()
        sample_mu = sample_mu
        if mnist == True:
            images = list(sample_mu.view(-1, 1, 28, 28).data.cpu())
        elif d3 == True:
            images = list(sample_mu.view(-1, 3, 64, 64).data.cpu())
        else:
            images = list(sample_mu.view(-1, 1, 64, 64).data.cpu())
        '''win_samples = vis.images(images, 10, 2, opts={'caption': 'samples'}, win=win_samples)'''
        save_image(images, 'img3.png')

    elif requirement == 'test_imgs':
        # plot the reconstructed distribution for the first 50 test images
        test_imgs = x[:50, :]
        _, reco_imgs, zs, _ = model.reconstruct_img(test_imgs)
        reco_imgs = reco_imgs.sigmoid()
        if mnist == True:
            test_reco_imgs = torch.cat([
                test_imgs.view(1, -1, 28, 28), reco_imgs.view(1, -1, 28, 28)], 0).transpose(0, 1)
        elif d3 == True:
            test_reco_imgs = torch.cat([
                test_imgs.view(3, -1, 64, 64), reco_imgs.view(3, -1, 64, 64)], 0).transpose(0, 1)
        else:
            test_reco_imgs = torch.cat([
                test_imgs.view(1, -1, 64, 64), reco_imgs.view(1, -1, 64, 64)], 0).transpose(0, 1)
        save_image(test_reco_imgs, 'img2.png')

        '''if mnist == True:
            win_test_reco = vis.images(
                list(test_reco_imgs.contiguous().view(-1, 1, 28, 28).data.cpu()), 10, 2,
                opts={'caption': 'test reconstruction image'}, win=win_test_reco)
        elif d3 == True:
            win_test_reco = vis.images(
                list(test_reco_imgs.contiguous().view(-1, 3, 64, 64).data.cpu()), 10, 2,
                opts={'caption': 'test reconstruction image'}, win=win_test_reco)
        else:
            win_test_reco = vis.images(
                list(test_reco_imgs.contiguous().view(-1, 1, 64, 64).data.cpu()), 10, 2,
                opts={'caption': 'test reconstruction image'}, win=win_test_reco)'''

    elif requirement == 'latent_walks':
        # plot latent walks (change one variable while all others stay the same)
        test_imgs = x[40000:40032, :].cuda()
        print("TEST IMGS", test_imgs.shape)
        '''test_imgs = test_imgs.unsqueeze(0)'''
        '''print("TEST_IMGS",test_imgs.shape)'''
        _, reco_imgs, zs, _ = model.reconstruct_img(test_imgs)
        print("ZS_INITIAL", zs.size(), zs[0:1].size())
        zs = zs[0:1]
        batch_size, z_dim = zs.size()
        print("ZS_SIZE", zs.size())
        xs = []
        delta = torch.autograd.Variable(torch.linspace(-1, 1, 7), volatile=True).type_as(zs)
        for i in range(z_dim):
            xs = []
            vec = Variable(torch.zeros(z_dim)).view(1, z_dim).expand(7, z_dim).contiguous().type_as(zs)
            vec[:, i] = 1
            vec = vec * delta[:, None]
            zs_delta = zs.clone().view(batch_size, 1, z_dim)
            zs_delta[:, :, i] = 0
            zs_walk = zs_delta + vec[None]
            xs_walk = model.decoder.forward(zs_walk.view(-1, z_dim)).sigmoid()
            xs.append(xs_walk)
            '''break'''
            print(len(xs))
            print(len(xs[0]))
            xs = list(torch.cat(xs, 0).data.cpu())
            print(len(xs))
            print(len(xs[0]))
            print(xs[0].shape)
            '''splitted = []'''
            '''win_latent_walk = vis.images(xs, 7, 2, opts={'caption': 'latent walk'}, win=win_latent_walk)'''
            for k in range(len(xs)):
                pad_size = (1,1,1,1)
                temp = F.pad(xs[k], pad_size, "constant", 255)
                save_image(temp, '3DShape_images/img1_3dshapes'+str(i)+'_'+str(k)+'.png')
                '''splitted.append(xs[k])
                splitted.append(torch.randn(3,64,64))'''
            '''transform = T.ToPILImage()
            img = transform(np.array(xs))
            img.save("img1_dSprites.png")'''
            '''print("I", str(i), 'img1_3dshapes'+str(i)+'.png')
            save_image(splitted, 'img1_3dshapes'+str(i)+'.png')'''



def only_random_display(model, mnist = False, d3 = False, mpi = False):
    # plot random samples
    sample_mu = model.model_sample(batch_size=100)
    sample_mu = sample_mu
    if mnist == True:
        images = list(sample_mu.view(-1, 1, 28, 28).data.cpu())
    elif d3 == True or mpi == True:
        print("TRUE")
        images = sample_mu.view(-1, 3, 64, 64).data.cpu()
        '''print(images.shape)'''
    else:
        print("HELLOIMAGES")
        images = list(sample_mu.view(-1, 1, 64, 64).data.cpu())
    '''win_samples = vis.images(images, 10, 2, opts={'caption': 'samples'}, win=win_samples)'''
    save_image(images, 'check_during_re/img3.png')
    '''print("IMAGES", images.shape)'''
    '''image = images[0]'''
    '''image = torch.permute(images, (1, 2, 0))
    print(image.shape)
    imgplot = plt.imshow(image)
    plt.savefig('img2.png')'''



def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-d', '--dataset', default='shapes', type=str, help='dataset name',
        choices=['shapes', 'faces', 'mnist', '3dshapes', 'mpi3D'])
    parser.add_argument('--checkpt', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save', type=str, default='.')
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    if args.dataset == 'mnist':
        mnist = True
    else:
        mnist = False

    if args.dataset == '3dshapes':
        d3 = True
    else:
        d3 = False

    if args.dataset == 'mpi3D':
        mpi = True
    else:
        mpi = False
    print("MPIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII", mpi)
    '''vae, dataset, cpargs = load_model_and_dataset(args.checkpt, only_model= False)
    print(len(dataset))
    vae.eval()
    display_samples(vae, dataset, mnist = mnist, d3 = d3, mpi = mpi)'''
    '''for i, x in enumerate(dataset):
        print(x.shape)
        x = x.cuda(non_blocking=True)
        test_imgs = x[:50, :]
        print(test_imgs.shape)
        vae.eval() 
        save_image(test_imgs, 'img_org.png')
        display_samples(vae, x, mnist = mnist, d3 = d3)
        break'''
    
    #vae, cpargs = load_model_and_dataset(args.checkpt, only_model= True)
    vae, dataset, cpargs = load_model_and_dataset(args.checkpt, only_model= False)
    only_random_display(vae, mnist = mnist, d3 = d3, mpi = mpi)


if __name__ == '__main__':
    model = main()

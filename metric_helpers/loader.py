import torch
import lib.dist as dist
import lib.flows as flows
#import vae_quant_cpy
#from vae_quant_cpy_integrated import VAE, setup_data_loaders
from vae_quant_mpi_inc import VAE, setup_data_loaders
#from models.vae import VAE
#from datasets.loading_datasets import setup_data_loaders

def load_model_and_dataset(checkpt_filename, test = True, only_model = False):
    print('Loading model and dataset.')
    checkpt = torch.load(checkpt_filename, map_location=lambda storage, loc: storage)
    args = checkpt['args']
    state_dict = checkpt['state_dict']

    # backwards compatibility
    if not hasattr(args, 'conv'):
        args.conv = False

    if not hasattr(args, 'dist') or args.dist == 'normal':
        prior_dist = dist.Normal()
        q_dist = dist.Normal()
        x_dist = dist.Normal()
    elif args.dist == 'laplace':
        prior_dist = dist.Laplace()
        q_dist = dist.Laplace()
        x_dist = dist.Laplace()
    elif args.dist == 'flow':
        prior_dist = flows.FactorialNormalizingFlow(dim=args.latent_dim, nsteps=32)
        q_dist = dist.Normal()
        x_dist = dist.Normal()

    # model
    if hasattr(args, 'ncon'):
        # InfoGAN
        model = infogan.Model(
            args.latent_dim, n_con=args.ncon, n_cat=args.ncat, cat_dim=args.cat_dim, use_cuda=True, conv=args.conv)
        model.load_state_dict(state_dict, strict=False)
        vae = vae_quant.VAE(
            z_dim=args.ncon, use_cuda=True, prior_dist=prior_dist, q_dist=q_dist, conv=args.conv)
        vae.encoder = model.encoder
        vae.decoder = model.decoder
    else:
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
        print("MPI LOADER",mpi)
        '''vae = vae_quant_cpy.VAE(
            z_dim=args.latent_dim, use_cuda=True, prior_dist=prior_dist, q_dist=q_dist,
        include_mutinfo=not args.exclude_mutinfo, tcvae=args.tcvae, conv=args.conv, mss=args.mss, mnist = mnist, d3 = d3)
        vae.load_state_dict(state_dict, strict=False)'''
        vae = VAE(
            z_dim=args.latent_dim, use_cuda=True, prior_dist=prior_dist, q_dist=q_dist, x_dist = x_dist,
        include_mutinfo=not args.exclude_mutinfo, tcvae=args.tcvae, conv=args.conv, mss=args.mss, mnist = mnist, d3 = d3, mpi = mpi, batch_size = 32)
        vae.load_state_dict(state_dict, strict=False)
    # dataset loader
    print(args.dataset)
    print("BATCH_SIZE", args.batch_size)
    #loader = vae_quant_cpy.setup_data_loaders(args, test = test)
    if only_model == False:
        loader = setup_data_loaders(args, test = test)
        print(len(loader.dataset))
        print(loader.dataset[0].shape)
        return vae, loader.dataset, args
    else:
        return vae, args

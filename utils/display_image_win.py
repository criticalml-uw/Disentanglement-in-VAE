import torch
from torch.autograd import Variable

win_samples = None
win_test_reco = None
win_latent_walk = None
win_train_elbo = None


def display_samples(model, x, vis, mnist = False, d3 = False):
    global win_samples, win_test_reco, win_latent_walk

    # plot random samples
    sample_mu = model.model_sample(batch_size=100).sigmoid()
    sample_mu = sample_mu
    if mnist == True:
        images = list(sample_mu.view(-1, 1, 28, 28).data.cpu())
    elif d3 == True:
        images = list(sample_mu.view(-1, 3, 64, 64).data.cpu())
    else:
        images = list(sample_mu.view(-1, 1, 64, 64).data.cpu())
    win_samples = vis.images(images, 10, 2, opts={'caption': 'samples'}, win=win_samples)

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

    if mnist == True:
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
            opts={'caption': 'test reconstruction image'}, win=win_test_reco)

    # plot latent walks (change one variable while all others stay the same)
    zs = zs[0:3]
    batch_size, z_dim = zs.size()
    xs = []
    delta = torch.autograd.Variable(torch.linspace(-2, 2, 7), volatile=True).type_as(zs)
    for i in range(z_dim):
        vec = Variable(torch.zeros(z_dim)).view(1, z_dim).expand(7, z_dim).contiguous().type_as(zs)
        vec[:, i] = 1
        vec = vec * delta[:, None]
        zs_delta = zs.clone().view(batch_size, 1, z_dim)
        zs_delta[:, :, i] = 0
        zs_walk = zs_delta + vec[None]
        xs_walk = model.decoder.forward(zs_walk.view(-1, z_dim)).sigmoid()
        xs.append(xs_walk)

    xs = list(torch.cat(xs, 0).data.cpu())
    win_latent_walk = vis.images(xs, 7, 2, opts={'caption': 'latent walk'}, win=win_latent_walk)
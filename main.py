from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import json
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np

import models.dcgan as dcgan
import models.cdcgan as cdcgan
import models.mlp as mlp

from dataset.simplex import SimplexDataset, UnlabeledSimplexDataset


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | simplex')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=3, help='input image channels')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
    parser.add_argument('--cuda_device', type=int, default=0, help='set cuda device')
    parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
    parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
    parser.add_argument('--mlp_G', action='store_true', help='use MLP for G')
    parser.add_argument('--mlp_D', action='store_true', help='use MLP for D')
    parser.add_argument('--conditional', action='store_true', help='train with conditions')
    parser.add_argument('--nfeatures', type=int, default=0, help='Number of conditional features')
    parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
    parser.add_argument('--experiment', default=None, help='Where to store samples and models')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
    opt = parser.parse_args()
    print(opt)

    if opt.experiment is None:
        opt.experiment = 'samples'
    os.system('mkdir {0}'.format(opt.experiment))

    opt.manualSeed = random.randint(1, 10000) # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if opt.dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = dset.ImageFolder(root=opt.dataroot,
                                transform=transforms.Compose([
                                    transforms.Scale(opt.imageSize),
                                    transforms.CenterCrop(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
    elif opt.dataset == 'lsun':
        dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                            transform=transforms.Compose([
                                transforms.Scale(opt.imageSize),
                                transforms.CenterCrop(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    elif opt.dataset == 'cifar10':
        dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                            transform=transforms.Compose([
                                transforms.Scale(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        )
    elif opt.dataset == 'simplex':
        dataset = SimplexDataset(root=opt.dataroot, isize=opt.imageSize)

    elif opt.dataset == 'unlabeledsimplex':
        dataset = UnlabeledSimplexDataset(root=opt.dataroot, isize=opt.imageSize)

    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                            shuffle=True, num_workers=int(opt.workers))

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    nc = int(opt.nc)
    nfeatures = int(opt.nfeatures)
    n_extra_layers = int(opt.n_extra_layers)

    # write out generator config to generate images together wth training checkpoints (.pth)
    generator_config = {"imageSize": opt.imageSize, "nz": nz, "nc": nc, "ngf": ngf, "ngpu": ngpu, "n_extra_layers": n_extra_layers, "nfeatures": nfeatures, "noBN": opt.noBN, "mlp_G": opt.mlp_G}
    with open(os.path.join(opt.experiment, "generator_config.json"), 'w') as gcfg:
        gcfg.write(json.dumps(generator_config)+"\n")

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    if opt.noBN:
        if opt.conditional:
            netG = cdcgan.DCGAN_G_nobn(opt.imageSize, nz, nc, ngf, ngpu, nfeatures, n_extra_layers)
        else:
            netG = dcgan.DCGAN_G_nobn(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)
    elif opt.mlp_G:
        netG = mlp.MLP_G(opt.imageSize, nz, nc, ngf, ngpu)
    else:
        if opt.conditional:
            netG = cdcgan.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu, nfeatures, n_extra_layers)
        else:
            netG = dcgan.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)

    # write out generator config to generate images together with training checkpoints (.pth)
    generator_config = {"imageSize": opt.imageSize, "nz": nz, "nc": nc, "ngf": ngf, "ngpu": ngpu, "nfeatures": nfeatures, "n_extra_layers": n_extra_layers, "noBN": opt.noBN, "mlp_G": opt.mlp_G}
    with open(os.path.join(opt.experiment, "generator_config.json"), 'w') as gcfg:
        gcfg.write(json.dumps(generator_config)+"\n")

    netG.apply(weights_init)
    if opt.netG != '': # load checkpoint if needed
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    if opt.mlp_D:
        netD = mlp.MLP_D(opt.imageSize, nz, nc, ndf, ngpu)
    elif opt.conditional:
        netD = cdcgan.DCGAN_D(opt.imageSize, nz, nc, ndf, ngpu, nfeatures, n_extra_layers)
        netD.apply(weights_init)
    else:
        netD = dcgan.DCGAN_D(opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers)
        netD.apply(weights_init)

    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    input = torch.FloatTensor(opt.batchSize, 4, opt.imageSize, opt.imageSize)
    noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
    fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
    
    if opt.dataset == 'simplex':
        forces = np.linspace(900, 1800, opt.batchSize)
        fixed_attr = np.column_stack([forces, forces, forces/2])
    else:
        fixed_attr = torch.FloatTensor(opt.batchSize, nfeatures)
    fixed_attr = torch.FloatTensor(fixed_attr).view(opt.batchSize, -1)
    
    one = torch.FloatTensor([1])
    mone = one * -1

    if opt.cuda:
        device = f'cuda:{opt.cuda_device}'
        netD.to(device)
        netG.to(device)
        input = input.to(device)
        one, mone = one.to(device), mone.to(device)
        noise, fixed_noise, fixed_attr = noise.to(device), fixed_noise.to(device), fixed_attr.to(device)

    # setup optimizer
    if opt.adam:
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
    else:
        optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lrD)
        optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG)

    errD_real_list = []
    errD_fake_list = []
    errD_list = []
    errG_list = []

    pbar = trange(opt.niter)

    gen_iterations = 0
    for epoch in pbar:
        data_iter = iter(dataloader)
        i = 0

        errD_real_cum = 0
        errD_fake_cum = 0
        errD_cum = 0
        errG_cum = 0

        while i < len(dataloader):
            ###########################
            # (1) Update D network
            ###########################
            for p in netD.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update

            # train the discriminator Diters times
            if gen_iterations < 25: # or gen_iterations % 500 == 0:
                Diters = 100
            else:
                Diters = opt.Diters
            j = 0
            while j < Diters and i < len(dataloader):
                j += 1

                # clamp parameters to a cube
                for p in netD.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

                data = data_iter.next()
                i += 1

                # train with real
                real_cpu, attr = data
                netD.zero_grad()
                batch_size = real_cpu.size(0)

                if opt.cuda:
                    real_cpu = real_cpu.to(device)
                    attr = attr.to(device)
                input.resize_as_(real_cpu).copy_(real_cpu)
                inputv = Variable(input)

                if opt.conditional:
                    errD_real = netD(inputv, attr)
                else:
                    errD_real = netD(inputv)
                errD_real.backward(one)

                # train with fake
                noise.resize_(attr.shape[0], nz, 1, 1).normal_(0, 1)
                noisev = Variable(noise, requires_grad=False) # totally freeze netG
                if opt.conditional:
                    fake = Variable(netG(noisev, attr).data)
                    errD_fake = netD(fake, attr)
                else:
                    fake = Variable(netG(noisev).data)
                    errD_fake = netD(fake)

                errD_fake.backward(mone)
                errD = errD_real - errD_fake
                optimizerD.step()

                errD_real_cum += errD_real.item() / Diters
                errD_fake_cum += errD_fake.item() / Diters
                errD_cum += errD.item() / Diters

            ############################
            # (2) Update G network
            ###########################
            for p in netD.parameters():
                p.requires_grad = False # to avoid computation
            netG.zero_grad()
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            noise.resize_(attr.shape[0], nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            
            if opt.conditional:
                fake = netG(noisev, attr)
                errG = netD(fake, attr)
            else:
                fake = netG(noisev)
                errG = netD(fake)
            errG.backward(one)
            optimizerG.step()

            errG_cum += errG.item()

            gen_iterations += 1

            #print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
            #    % (epoch, opt.niter, i, len(dataloader), gen_iterations,
            #    errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))
            
            pbar.set_description('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                  % (epoch, opt.niter, i, len(dataloader), gen_iterations,
                     errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))

            #pbar.set_description('[%d/%d] Loss_D: %.4f Loss_G: %.4f'
            #      % (i, len(dataloader), errD.item(), errG.item()))
            if gen_iterations % 1 == 0:
                real_cpu = real_cpu.mul(0.5).add(0.5)
                vutils.save_image(real_cpu, '{0}/real_samples.png'.format(opt.experiment))
                
                if opt.conditional:
                    fake = netG(Variable(fixed_noise), fixed_attr)
                else:
                    fake = netG(Variable(fixed_noise))
                fake.data = fake.data.mul(0.5).add(0.5)
                vutils.save_image(fake.data, '{0}/fake_samples_{1}.png'.format(opt.experiment, gen_iterations))

        # do checkpointing
        torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
        torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))

        errD_real_list.append(errD_real_cum / len(dataloader))
        errD_fake_list.append(errD_fake_cum / len(dataloader))
        errD_list.append(errD_cum / len(dataloader))
        errG_list.append(errG_cum / len(dataloader))

        if epoch % 50 == 0 or epoch == opt.niter - 1:
            # real vs fake discriminator loss
            plt.figure()
            plt.plot(range(epoch + 1), errD_real_list, label="real")
            plt.plot(range(epoch + 1), errD_fake_list, label="fake")
            plt.xlabel("Epochs")
            plt.ylabel("Discriminator loss")
            plt.legend(loc='best')
            plt.savefig(f'{opt.experiment}/netD_loss_epoch_{epoch}.png')

            # discriminator vs generator loss
            plt.figure()
            plt.plot(range(epoch + 1), errD_list, label="netD")
            plt.plot(range(epoch + 1), errG_list, label="netG")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend(loc='best')
            plt.savefig(f'{opt.experiment}/loss_epoch_{epoch}.png')

            # discriminator vs generator loss
            plt.figure()
            plt.plot(range(epoch + 1), np.abs(errD_list))
            plt.xlabel("Generator epochs")
            plt.ylabel("Wasserstein estimate")
            plt.savefig(f'{opt.experiment}/EM_epoch_{epoch}.png')

            # real vs fake surfaces
            # real
            #real = dataset[np.random.randint(len(dataset), size=3)]
            # fake
            #noise = torch.randn(3, nz, 1, 1)
            #fake = netG(noise.cuda())
            #data = torch.cat((real, fake.cpu()))
            #grid_img = vutils.make_grid(data, nrow=3)
            #plt.imshow(grid_img.permute(1, 2, 0), cmap='Greys')
            #plt.ylabel("Fake   |   Real")
            #plt.colorbar()
            # plt.axis('off')
            #plt.savefig(f"{opt.experiment}/real_fake_{epoch}.png")

            # check periodicity
            noise = torch.randn(1, nz, 1, 1, device=device)
            if opt.conditional:
                attr = torch.FloatTensor([1600, 1600, 800])
                if opt.cuda:
                    attr = attr.to(device)
                fake = netG(noise, attr)
            else:
                fake = netG(noise)
            fake_numpy = fake.cpu().detach().numpy()[0, 0]
            fake_tiled = np.tile(fake_numpy, (2, 2))
            plt.figure()
            plt.imshow(fake_tiled, cmap='Greys')
            plt.axis('off')
            plt.colorbar()
            plt.savefig(f'{opt.experiment}/periodic_{epoch}.png')

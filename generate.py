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

import models.dcgan as dcgan
import models.cdcgan as cdcgan
import models.mlp as mlp

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, type=str, help='path to generator config .json file')
    parser.add_argument('-w', '--weights', required=True, type=str, help='path to generator weights .pth file')
    parser.add_argument('-o', '--output_dir', required=True, type=str, help="path to to output directory")
    parser.add_argument('-n', '--nimages', required=True, type=int, help="number of images to generate", default=1)
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--cuda_device', type=int, default=0, help='GPU id')
    parser.add_argument('--conditional', action='store_true', help='conditional input')
    parser.add_argument('--attr', type=float, default=[], nargs="+", help='attributes')
    opt = parser.parse_args()

    with open(opt.config, 'r') as gencfg:
        generator_config = json.loads(gencfg.read())
    
    imageSize = generator_config["imageSize"]
    nz = generator_config["nz"]
    nc = generator_config["nc"]
    ngf = generator_config["ngf"]
    noBN = generator_config["noBN"]
    ngpu = generator_config["ngpu"]
    mlp_G = generator_config["mlp_G"]
    n_extra_layers = generator_config["n_extra_layers"]

    if opt.conditional:
        nfeatures = generator_config["nfeatures"]
        if noBN:
            netG = cdcgan.DCGAN_G_nobn(imageSize, nz, nc, ngf, ngpu, nfeatures, n_extra_layers)
        else:
            netG = cdcgan.DCGAN_G(imageSize, nz, nc, ngf, ngpu, nfeatures, n_extra_layers)
    elif mlp_G:
        netG = mlp.MLP_G(imageSize, nz, nc, ngf, ngpu)
    else:
        if noBN:
            netG = dcgan.DCGAN_G_nobn(imageSize, nz, nc, ngf, ngpu, n_extra_layers)
        else:
            netG = dcgan.DCGAN_G(imageSize, nz, nc, ngf, ngpu, n_extra_layers)

    # load weights
    netG.load_state_dict(torch.load(opt.weights))

    # initialize noise
    fixed_noise = torch.FloatTensor(opt.nimages, nz, 1, 1).normal_(0, 1)
    fixed_attr = torch.tile(torch.FloatTensor(opt.attr), (opt.nimages,))
    print(fixed_attr)

    if opt.cuda:
        device = f'cuda:{opt.cuda_device}'
        netG.to(device)
        fixed_noise = fixed_noise.to(device)
        fixed_attr = fixed_attr.to(device)

    if opt.conditional:
        fake = netG(fixed_noise, fixed_attr)
    else:
        fake = netG(fixed_noise)
    fake.data = fake.data.mul(0.5).add(0.5)

    for i in range(opt.nimages):
        vutils.save_image(fake.data[i, ...].reshape((1, nc, imageSize, imageSize)), os.path.join(opt.output_dir, "generated_%02d.png"%i))

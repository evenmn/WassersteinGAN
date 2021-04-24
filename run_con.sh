#!/bin/bash
python3 main.py --dataset=simplex --dataroot=/home/users/evenmn/ml-friction/dataset/ --cuda --cuda_device=0 --ngpu=1 --conditional --nfeatures=3 --nc=1 --nz=20 --ndf=16 --ngf=16 --imageSize=128 --experiment=wcdcgan/run_test5 --niter=100000 --lrD=5e-5 --lrG=5e-5 --netD=wcdcgan/run_test4/netD_epoch_9999.pth --netG=wcdcgan/run_test4/netG_epoch_9999.pth

#!/bin/bash
python3 main.py --dataset=simplex --dataroot=/home/users/evenmn/ml-friction/dataset/ --cuda --ngpu=1 --cuda_device=4 --nc=1 --nz=50 --ndf=16 --ngf=16 --imageSize=128 --experiment=wdcgan/run_nz_50 --niter=10000 --lrD=5e-5 --lrG=5e-5

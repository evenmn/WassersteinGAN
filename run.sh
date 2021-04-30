#!/bin/bash
python3 main.py --dataset=simplex --dataroot=/home/users/evenmn/ml-friction/dataset/ --cuda --ngpu=1 --cuda_device=2 --nc=1 --nz=20 --ndf=16 --ngf=16 --imageSize=128 --experiment=wdcgan/run_test --niter=1 --lrD=5e-5 --lrG=5e-5

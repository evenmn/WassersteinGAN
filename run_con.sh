#!/bin/bash
python3 main.py --dataset=simplex --dataroot=/home/users/evenmn/ml-friction/dataset/ --cuda --cuda_device=0 --ngpu=1 --conditional --nfeatures=3 --nc=1 --nz=20 --ndf=16 --ngf=16 --imageSize=128 --experiment=wcdcgan/run_test8 --niter=30000 --lrD=1e-4 --lrG=1e-4

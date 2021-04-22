#!/bin/bash
python3 main.py --dataset=simplex --dataroot=/home/users/evenmn/ml-friction/dataset/ --cuda --ngpu=1 --conditional --nfeatures=3 --nc=1 --nz=20 --ndf=16 --ngf=16 --imageSize=128 --experiment=wcdcgan/run_test --niter=10000 --lrD=5e-5 --lrG=5e-5

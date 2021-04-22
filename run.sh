#!/bin/bash
python3 main.py --dataset=simplex --dataroot=/ --cuda --ngpu=1 --nc=1 --nz=20 --ndf=16 --ngf=16 --imageSize=128 --experiment=wdcgan/run_periodic3 --niter=2000 --lrD=1e-4 --lrG=1e-4

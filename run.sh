#!/bin/bash
python3 main.py --dataset=simplex --dataroot=/ --cuda --adam --nc=1 --nz=20 --ndf=16 --ngf=16 --imageSize=128 --experiment=/datastorage/ml-friction/wdcgan/run_periodic --niter=2000 --lrD=5e-5 --lrG=5e-5

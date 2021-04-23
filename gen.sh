#!/bin/bash

python3 generate.py --config=wdcgan/run_periodic6/generator_config.json --weights=wdcgan/run_periodic6/netG_epoch_999.pth --output_dir=wdcgan/gen_periodic6 --nimages=5 --cuda

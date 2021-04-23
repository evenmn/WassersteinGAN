#!/bin/bash

python3 generate.py --config=wcdcgan/run_test4/generator_config.json --weights=wcdcgan/run_test4/netG_epoch_6000.pth --output_dir=wcdcgan/gen_test4 --nimages=5 --conditional --cuda --attr 1600 1600 800 

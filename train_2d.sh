#!/bin/bash
python train.py --slicing xy -bs 32
python train.py --slicing yz -bs 16
python train.py --slicing zx -bs 16
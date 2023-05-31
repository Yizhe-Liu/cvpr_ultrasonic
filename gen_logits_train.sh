#!bin/bash
for idx in {1..7}
do
    python predict.py --ckpt trained/xy.ckpt --slicing=xy --idx $idx --label=soft
    python predict.py --ckpt trained/yz.ckpt --slicing=yz --idx $idx --label=soft
    python predict.py --ckpt trained/zx.ckpt --slicing=zx --idx $idx --label=soft
done
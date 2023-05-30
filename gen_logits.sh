#!bin/bash
for idx in {1..7}
do
    python predict.py --ckpt pretrained/xy.ckpt --slicing=xy --idx $idx --label=soft
    python predict.py --ckpt pretrained/yz.ckpt --slicing=yz --idx $idx --label=soft
    python predict.py --ckpt pretrained/zx.ckpt --slicing=zx --idx $idx --label=soft
done
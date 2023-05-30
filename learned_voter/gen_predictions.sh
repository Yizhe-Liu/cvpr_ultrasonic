#!bin/bash
for idx in {1..7}
do
    python predict.py --ckpt ../pretrained/voter.ckpt --idx $idx 
done
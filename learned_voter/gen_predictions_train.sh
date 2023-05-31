#!bin/bash
for idx in {1..89}
do
    python predict.py --ckpt ../pretrained/voter.ckpt --idx $idx 
done
#!bin/bash
for idx in {1..5}
do
    python predict.py --ckpt lightning_logs/version_6/checkpoints/epoch\=32-step\=4752.ckpt --idx $idx --label=soft
    python predict.py --ckpt lightning_logs/version_7/checkpoints/epoch\=37-step\=5472.ckpt --slicing=yz  --idx $idx --label=soft
    python predict.py --ckpt lightning_logs/version_8/checkpoints/epoch\=22-step\=3312.ckpt --slicing=zx  --idx $idx --label=soft
done
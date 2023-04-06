import pytorch_lightning as pl
import torch
import os
from model import Simple3DCNN
from argparse import ArgumentParser
from data import SlicingLogits
from torch.utils.data import DataLoader
from itertools import product


def train(path, ckpt_path, idx, grid_len, out_dir):
    pl.seed_everything(0)
    ds = SlicingLogits(path, grid_len, idx)
    dl = DataLoader(ds, batch_size=1)
    model = Simple3DCNN()
    trainer = pl.Trainer(accelerator='gpu', precision=16)
    pred = trainer.predict(model, dl, ckpt_path=ckpt_path)
    occ = torch.zeros((1280, 768, 768), dtype=bool)
    for (i, j, k), cube in \
        zip(product(range(0, 1280, grid_len), range(0, 768, grid_len), range(0, 768, grid_len)), 
            pred):
        occ[i:i+grid_len, j:j+grid_len, k:k+grid_len] = cube > 0

    torch.save(occ, os.path.join(out_dir, f'pred_{idx:03}_voted.pt'))


if __name__ == '__main__':
    parser = ArgumentParser("Learnt Voter")
    parser.add_argument('--path', '-p', default='data/', help='Dataset path')
    parser.add_argument('--ckpt_path', '-c', help='Checkpoint path')
    parser.add_argument('--idx', type=int, default=5, help='Scan Index')
    parser.add_argument('--grid_len', type=int, default=256, help='Grid Length')
    parser.add_argument('--out_dir', type=str, default='../output', help='Output path')

    args = parser.parse_args()
    train(args.path, args.ckpt_path, args.idx, args.grid_len, args.out_dir)
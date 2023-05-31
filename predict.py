import os
import torch
import pytorch_lightning as pl
from data import UT2DRawOnlyDataset, UT3DRawOnlyDataset
from model import UNet2DModel, UNet3DModel
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from itertools import product, chain


def predict(model_dim, layers, channels, slicing, path, bs, ckpt_path, idx, out_dir, label, grid_len=128):
    pl.seed_everything(0)
    if model_dim == 2:
        ds = UT2DRawOnlyDataset(path, idx, slicing, channels)
        dl = DataLoader(ds, bs, shuffle=False, num_workers=20)
        model = UNet2DModel(channels, layers)

    if model_dim == 3:
        ds = UT3DRawOnlyDataset(path, idx, grid_len)
        dl = DataLoader(ds, bs, shuffle=False, num_workers=20)
        model = UNet3DModel()

    trainer = pl.Trainer(accelerator='gpu', devices=1, precision=16)
    pred = torch.vstack(trainer.predict(model, dl, ckpt_path=ckpt_path))

    if label == 'hard':
            pred = pred > 0

    if model_dim == 2:
        pred = torch.nn.functional.pad(pred, [0, 0, 0, 0, channels//2, channels//2])
        if slicing == 'yz':
            pred = pred.permute(2, 0, 1)
        elif slicing == 'zx':
            pred = pred.permute(1, 2, 0)
        
        torch.save(pred, os.path.join(out_dir, f'pred_{idx:03}_{slicing}_{model_dim}d_{label}_{layers}layers.pt'))

    if model_dim == 3:
        occ = torch.zeros((1280, 768, 768), dtype=bool)
        for (i, j, k), cube in \
            zip(product(range(0, 1280, grid_len), range(0, 768, grid_len), range(0, 768, grid_len)), 
                pred):
            occ[i:i+grid_len, j:j+grid_len, k:k+grid_len] = cube

        torch.save(occ, os.path.join(out_dir, f'pred_{idx:03}_{model_dim}d_{label}.pt'))

if __name__ == '__main__':
    parser = ArgumentParser("CNN Predictor")
    parser.add_argument('--model_dim', type=int, default=2, choices=[2, 3], help='Model Dimension')    
    parser.add_argument('--layers', type=int, default=5, help='Model Layers')
    parser.add_argument('--channels', type=int, default=5, help='2D CNN Input Channels')
    parser.add_argument('--slicing', type=str, default='xy', choices=['xy', 'yz', 'zx'], help='2D CNN slicing direction')
    parser.add_argument('--path', '-p', default='data/', help='Dataset path')
    parser.add_argument('--out_dir', '-o', default='output/', help='Output path')
    parser.add_argument('--ckpt', '-c', help='Checkpoint path')
    parser.add_argument('--batch_size', '-bs', type=int, default=40, help='Batch size')
    parser.add_argument('--idx', type=int, default=5, help='Index of the scans.')
    parser.add_argument('--label', choices=['soft', 'hard'], default='hard', help='Soft label (Logits) or Hard Label (Binary)')
    
    args = parser.parse_args()
    predict(args.model_dim, args.layers, args.channels, args.slicing, args.path, args.batch_size, args.ckpt, args.idx, args.out_dir, args.label)
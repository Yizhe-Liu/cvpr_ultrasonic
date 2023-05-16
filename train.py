import torch
import pytorch_lightning as pl
from model import UNet2DModel, UNet3DModel
from argparse import ArgumentParser
from data import UT2DDataModule, UT3DDataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def train(model_dim, channels, slicing, path, bs, max_ep, ckpt_path=None, semi_supervised=False):
    pl.seed_everything(0)
    if semi_supervised:
        train_indices = [f'{i:03}' for i in range(1, 6)]
    else:
        train_indices = [f'{i:03}' for i in range(1, 6)]

    if model_dim == 2:
        dl = UT2DDataModule(path, bs, channels, slicing, train_indices)
        model = UNet2DModel(channels)

    if model_dim == 3:
        dl = UT3DDataModule(path, bs, train_indices)
        model = UNet3DModel()
    
    #compiled_model = torch.compile(model)

    trainer = pl.Trainer(accelerator='gpu', max_epochs=max_ep, precision=16, log_every_n_steps=1,
                         callbacks=[EarlyStopping(monitor="train_loss", mode="min", patience=10)])
    if ckpt_path:
        trainer.fit(model, dl, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, dl)



if __name__ == '__main__':
    parser = ArgumentParser("CNN Trainer")
    parser.add_argument('--model_dim', type=int, default=2, choices=[2, 3], help='Model Dimension')
    parser.add_argument('--channels', type=int, default=5, help='2D CNN Input Channels')
    parser.add_argument('--slicing', type=str, default='xy', choices=['xy', 'yz', 'zx'], help='2D CNN slicing direction')
    parser.add_argument('--path', '-p', default='data/', help='Dataset path')
    parser.add_argument('--ckpt_path', '-c', help='Checkpoint path')
    parser.add_argument('--epochs', type=int, default=100, help='Max epoches')
    parser.add_argument('--batch_size', '-bs', type=int, default=8, help='Batch size')
    parser.add_argument('--semi_supervised',  action='store_true', help='Use semi supervised training')

    args = parser.parse_args()
    train(args.model_dim, args.channels, args.slicing, args.path, args.batch_size, args.epochs, args.ckpt_path, args.semi_supervised)
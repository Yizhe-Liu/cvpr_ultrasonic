import pytorch_lightning as pl
from model import Simple3DCNN
from argparse import ArgumentParser
from data import SlicingLogitsDM
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def train(path, bs, max_ep, ckpt_path=None):
    pl.seed_everything(0)
    dl = SlicingLogitsDM(path, bs)
    model = Simple3DCNN()

    trainer = pl.Trainer(accelerator='gpu', max_epochs=max_ep, precision=16, log_every_n_steps=1,
                         callbacks=[EarlyStopping(monitor="train_loss", mode="min", patience=5),
                                    ModelCheckpoint('../trained/', 'voter')])
    if ckpt_path:
        trainer.fit(model, dl, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, dl)



if __name__ == '__main__':
    parser = ArgumentParser("Learnt Voter")
    parser.add_argument('--path', '-p', default='../output', help='Dataset path')
    parser.add_argument('--ckpt_path', '-c', help='Checkpoint path')
    parser.add_argument('--epochs', type=int, default=100, help='Max epoches')
    parser.add_argument('--grid_len', type=int, default=64, help='Grid length')
    parser.add_argument('--batch_size', '-bs', type=int, default=32, help='Batch size')

    args = parser.parse_args()
    train(args.path, args.batch_size, args.epochs, args.ckpt_path)
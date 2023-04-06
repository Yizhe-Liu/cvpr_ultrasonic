import pytorch_lightning as pl
from torch import optim, nn
from torch.nn import functional as F

class Simple3DCNN(pl.LightningModule):
    def __init__(self, channels=3):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(channels, 8, 3, padding='same'), nn.GELU(), nn.Conv3d(8, 8, 3, padding='same'), nn.GELU(), nn.Conv3d(8, 1, 3, padding='same'))
        self.loss = nn.BCEWithLogitsLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        return self.net(batch[0])[:, 0] 

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
import pytorch_lightning as pl
from torch import optim, nn, Tensor
from pl_bolts.models.vision import UNet
import torch
from torch.nn import functional as F

class UNet2DModel(pl.LightningModule):
    def __init__(self, channels=1, num_layers=5):
        super().__init__()
        self.unet = UNet(1, channels, num_layers, features_start=16)
        self.loss = nn.BCEWithLogitsLoss()
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.unet(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.unet(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.unet(x)
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        return self.unet(batch)[:, 0] 

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    


class UNet3D(nn.Module):
    """Based on Pytorch Lightning implementation of U-Net.
    Paper: `U-Net: Convolutional Networks for Biomedical Image Segmentation
    <https://arxiv.org/abs/1505.04597>`_
    Paper authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox
    Implemented by:
        - `Annika Brundyn <https://github.com/annikabrundyn>`_
        - `Akshay Kulkarni <https://github.com/akshaykvnit>`_
    Args:
        num_classes: Number of output classes required
        input_channels: Number of channels in input images (default 3)
        num_layers: Number of layers in each side of U-net (default 5)
        features_start: Number of features in first layer (default 64)
        bilinear: Whether to use bilinear interpolation (True) or transposed convolutions (default) for upsampling.
    """

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        num_layers: int = 5,
        features_start: int = 16,
        bilinear: bool = False,
    ):

        if num_layers < 1:
            raise ValueError(f"num_layers = {num_layers}, expected: num_layers > 0")

        super().__init__()
        self.num_layers = num_layers

        layers = [DoubleConv3D(input_channels, features_start)]

        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Down3D(feats, feats*3))
            feats *= 3

        for _ in range(num_layers - 1):
            layers.append(Up3D(feats, feats//3, bilinear))
            feats //= 3

        layers.append(nn.Conv3d(feats, num_classes, kernel_size=1))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1 : self.num_layers]:
            xi.append(layer(xi[-1]))
        # Up path
        for i, layer in enumerate(self.layers[self.num_layers : -1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        return self.layers[-1](xi[-1])


class DoubleConv3D(nn.Module):
    """[ Conv3d => BatchNorm => ReLU ] x 2."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Down3D(nn.Module):
    """Downscale with MaxPool => DoubleConvolution block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool3d(kernel_size=2, stride=2), DoubleConv3D(in_ch, out_ch))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Up3D(nn.Module):
    """Upsampling (by either bilinear interpolation or transpose convolutions) followed by concatenation of feature
    map from contracting path, followed by DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv3d(in_ch, in_ch // 3, kernel_size=1),
            )
        else:
            self.upsample = nn.ConvTranspose3d(in_ch, in_ch // 3, kernel_size=2, stride=2)

        self.conv = DoubleConv3D(in_ch // 3*2, out_ch)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_d = x2.shape[2] - x1.shape[2]
        diff_h = x2.shape[3] - x1.shape[3]
        diff_w = x2.shape[4] - x1.shape[4]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2, diff_d // 2, diff_d - diff_d // 2])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

            

class ConnectivityLoss3D(nn.Module):
    def __init__(self, weight=0.0001):
        super().__init__()
        self.supervised_loss = nn.BCEWithLogitsLoss()
        self.neighbours = torch.cartesian_prod(*[Tensor([-1, 0, 1]).type(torch.int)]*3)
        self.w = weight
    
    def forward(self, x, y, is_gt):
        loss = Tensor([0]).cuda()
        for _x, _y, _is_gt in zip(x, y, is_gt):
            pred = (_x > 0).type(torch.int8) # logits to label
            acc = sum([torch.roll(pred, s.tolist(), (1,2,3)) for s in self.neighbours])
            loss += _is_gt*self.supervised_loss(_x, _y) + self.w*(acc==pred).float().mean()

        return loss
    


class ConnectivityLoss2D(nn.Module):
    def __init__(self, weight=0.001):
        super().__init__()
        self.supervised_loss = nn.BCEWithLogitsLoss()
        self.neighbours = torch.cartesian_prod(*[torch.Tensor([-1, 0, 1]).type(torch.int)]*2)
        self.w = weight
    
    def forward(self, x, y, is_gt):
        loss = 0
        for _x, _y, _is_gt in zip(x, y, is_gt):
            pred = (_x > 0).type(torch.int8) # logits to label
            acc = sum([torch.roll(pred, s.tolist(), (1,2)) for s in self.neighbours])
            loss += _is_gt*self.supervised_loss(_x, _y) + self.w*(acc==pred).mean()

        return loss
    

class UNet3DModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.unet = UNet3D(1, 1, 5, features_start=16)
        self.loss = ConnectivityLoss3D()
    
    def training_step(self, batch, batch_idx):
        x, y, is_gt = batch
        y_hat = self.unet(x)
        loss = self.loss(y_hat, y, is_gt)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, is_gt = batch
        y_hat = self.unet(x)
        loss = self.loss(y_hat, y, is_gt)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, is_gt = batch
        y_hat = self.unet(x)
        loss = self.loss(y_hat, y, is_gt)
        self.log("test_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        return self.unet(batch)[:, 0]

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
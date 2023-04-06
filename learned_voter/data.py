import pytorch_lightning as pl
import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split


class SlicingLogits(Dataset):
    def __init__(self, path, side_len=64, indices=None):
        if not indices:
            indices = [f'{i:03}' for i in range(1, 6)]
        if type(indices) is int:
            indices = [f'{indices:03}']
        if type(indices) is str:
            indices = [indices]
            
        self.scans = []
        self.oc_fs = []
        for idx in indices:
            p = os.path.join(path, f'gt_{idx}.npy')
            gt = torch.from_numpy(np.load(p).astype(np.float16)).permute(2, 0, 1)
            logits = []
            for s in ['xy', 'yz', 'zx']:
                p = os.path.join(path, f'pred_{idx}_{s}_2d_soft_5layers.pt')
                logits.append(torch.load(p))
            
            logits = torch.stack(logits) # 3*1280*768*768
            c, h, w, d = logits.shape
            for i in range(0, h, side_len):
                for j in range(0, w, side_len):
                    for k in range(0, d, side_len):
                        self.scans.append(logits[:, i:i+side_len, j:j+side_len, k:k+side_len])
                        self.oc_fs.append(gt[None, i:i+side_len, j:j+side_len, k:k+side_len])


    def __len__(self):
        return len(self.scans)
    
    def __getitem__(self, idx):
        return self.scans[idx], self.oc_fs[idx]
    

class SlicingLogitsDM(pl.LightningDataModule):
    def __init__(self, path, batch_size, side_len=64, train_indices=None):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.side_len = side_len
        self.train_indices = train_indices
        

    def prepare_data(self):
        self.ds = SlicingLogits(self.path, self.side_len, self.train_indices)

    def setup(self, stage):
        n = len(self.ds)
        self.train, self.valid = random_split(self.ds, [int(n*0.9), n-int(n*0.9)])

    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, shuffle=True, num_workers=20)
    
    def val_dataloader(self):
        return DataLoader(self.valid, self.batch_size, shuffle=False, num_workers=20)
    
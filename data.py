import pytorch_lightning as pl
import torch
import numpy as np
import os
from random import randint
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from itertools import product

class UT2DDataset(Dataset):
    def __init__(self, path, indices=None, slicing='xy', channels=1):
        if not indices:
            indices = [f'{i:03}' for i in range(1, 6)]

        data_files = [os.path.join(path, 'volumes', f'scan_{i}.raw') for i in indices]
        occ_files = [os.path.join(path, 'occ_field', f'gt_{i}.npy') for i in indices]
        self.scans = []
        self.oc_fs = []
        self.channels = channels

        for df, of in tqdm(zip(data_files, occ_files), 'Loading Dataset', total=len(data_files)):
            # load scan in zxy order
            scan = np.fromfile(df, '<u2').reshape(1280, 768, 768)/32768
            # load occupancy field
            oc_f = np.ascontiguousarray(np.transpose(np.load(of), [2, 0, 1]))
            
            if slicing == 'yz':
                scan = np.transpose(scan, [1, 2, 0])
                oc_f = np.transpose(oc_f, [1, 2, 0])

            elif slicing == 'zx':
                scan = np.transpose(scan, [2, 0, 1])
                oc_f = np.transpose(oc_f, [2, 0, 1])
                

            self.scans.append(scan)
            self.oc_fs.append(oc_f)


        self.scans = torch.from_numpy(np.vstack(self.scans).astype(np.float16))
        self.oc_fs = torch.from_numpy(np.vstack(self.oc_fs).astype(np.float16))


    def __len__(self):
        return len(self.scans) - self.channels + 1
    
    def __getitem__(self, idx):
        return self.scans[idx:idx+self.channels], self.oc_fs[idx + self.channels//2][None,:]
        

class UT2DRawOnlyDataset(Dataset):
    def __init__(self, path, idx, slicing, channels):
        f = os.path.join(path, 'volumes', f'scan_{idx:03}.raw')
        self.scans = (np.fromfile(f, '<u2').reshape(1280, 768, 768)/32768.0).astype(np.float16)

        if slicing == 'yz':
            self.scans = np.transpose(self.scans, [1, 2, 0])

        elif slicing == 'zx':
            self.scans = np.transpose(self.scans, [2, 0, 1])
                
        self.channels = channels

    def __len__(self):
        return len(self.scans) - self.channels + 1
    
    def __getitem__(self, idx):
        return self.scans[idx:idx+self.channels]
           

class UT2DDataModule(pl.LightningDataModule):
    def __init__(self, path, batch_size, channels, slicing, train_indices=None):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.train_indices = train_indices
        self.channels = channels
        self.slicing = slicing
        

    def prepare_data(self):
        self.ds = UT2DDataset(self.path, self.train_indices, self.slicing, self.channels)

    def setup(self, stage):
        n = len(self.ds)
        self.train, self.valid = random_split(self.ds, [int(n*0.9), n-int(n*0.9)])

    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, shuffle=True, num_workers=2)
    
    def val_dataloader(self):
        return DataLoader(self.valid, self.batch_size, shuffle=False, num_workers=2)
    

class UT3DDataset(Dataset):
    def __init__(self, path, indices=None, grid_len=128):
        if not indices:
            indices = [f'{i:03}' for i in range(1, 6)]

        self.grid_len = grid_len
        data_files = [os.path.join(path, 'volumes', f'scan_{i}.raw') for i in indices]
        occ_files = [os.path.join(path, 'occ_field', f'gt_{i}.npy') for i in indices]
        n = len(data_files)
        self.scans = torch.zeros((n, 1280, 768, 768), dtype=torch.float16)
        self.oc_fs = torch.zeros((n, 1280, 768, 768), dtype=torch.bool)
        self.is_gt = []

        self.n = len(self.scans)*1280*768**2//grid_len**3

        for i, (df, of) in enumerate(tqdm(zip(data_files, occ_files), 'Loading Datset', total=len(data_files))):
            # load scan
            self.scans[i] = torch.from_numpy(np.fromfile(df, '<u2').reshape(1280, 768, 768)/32768)
            if os.path.isfile(of):
                self.oc_fs[i] = torch.from_numpy(np.transpose(np.load(of), [2, 0, 1]))
                self.is_gt.append(True)
            else:
                self.is_gt.append(False)


    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        i = randint(0, len(self.scans)-1)
        grid_len = self.grid_len
        x, y, z = randint(0, 1280 - grid_len - 1), randint(0, 768 - grid_len - 1), randint(0, 768 - grid_len - 1)
        scan = self.scans[i,None,x:x+grid_len,y:y+grid_len,z:z+grid_len]
        oc_f = self.oc_fs[i,None,x:x+grid_len,y:y+grid_len,z:z+grid_len].type(torch.float16)

        return scan, oc_f, self.is_gt[i]


class UT3DDataModule(pl.LightningDataModule):
    def __init__(self, path, batch_size, train_indices=None):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.train_indices = train_indices

    def prepare_data(self):
        self.ds = UT3DDataset(self.path, self.train_indices)

    def setup(self, stage):
        n = len(self.ds)
        self.train, self.valid = random_split(self.ds, [int(n*0.9), n-int(n*0.9)])

    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, shuffle=True, num_workers=20)
    
    def val_dataloader(self):
        return DataLoader(self.valid, self.batch_size, shuffle=False, num_workers=20)


class UT3DRawOnlyDataset(Dataset):
    def __init__(self, path, idx, grid_len=128):
        f = os.path.join(path, 'volumes', f'scan_{idx:03}.raw')
        raw = np.fromfile(f, '<u2').reshape(1280, 768, 768).astype(np.float16)/32768
        self.scans = []
        for i in range(0, 1280, grid_len):
            for j in range(0, 768, grid_len):
                for k in range(0, 768, grid_len):
                    self.scans.append(raw[i:i+grid_len, j:j+grid_len, k:k+grid_len])

    def __len__(self):
        return len(self.scans)
    
    def __getitem__(self, idx):
        return self.scans[idx][None, :]
           

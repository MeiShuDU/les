import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SyntheticWaterDataset(Dataset):
    def __init__(self, n_samples=100, n_atoms=6, box_len=10.0, seed=42):
        super().__init__()
        self.n_samples = n_samples
        self.n_atoms = n_atoms
        self.box_len = box_len
        torch.manual_seed(seed)
        
        # 随机生成坐标（在盒子内）
        coords = torch.rand(n_samples, n_atoms, 3) * box_len
        # 原子类型固定（水分子顺序：O H H O H H）
        atype = torch.tensor([0, 1, 1, 0, 1, 1], dtype=torch.int32).repeat(n_samples, 1)
        # 晶胞（对角矩阵）
        cell = torch.eye(3) * box_len
        cell = cell.unsqueeze(0).repeat(n_samples, 1, 1)
        
        # 随机生成能量和受力（仅用于测试）
        energy = -100.0 + 50.0 * torch.rand(n_samples, 1)
        force = 10.0 * torch.rand(n_samples, n_atoms, 3) - 5.0  # 范围 -5 ~ 5
        
        self.data = {
            'coord': coords,
            'atype': atype,
            'cell': cell,
            'energy': energy,
            'force': force,
        }

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            'coord': self.data['coord'][idx],
            'atype': self.data['atype'][idx],
            'cell': self.data['cell'][idx],
            'energy': self.data['energy'][idx],
            'force': self.data['force'][idx],
        }
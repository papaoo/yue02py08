import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple


class PathPlanningDataset(Dataset):
    def __init__(self, data_dir: str, split: str = 'train'):
        self.data_dir = os.path.join(data_dir, split)
        self.split = split
        
        input_files = [f for f in os.listdir(self.data_dir) if f.startswith('input_')]
        self.num_samples = len(input_files)
        
        if self.num_samples == 0:
            raise ValueError(f"在 {self.data_dir} 中未找到数据文件")
        
        print(f"加载 {split} 数据集，共 {self.num_samples} 个样本")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_path = os.path.join(self.data_dir, f'input_{idx}.npy')
        label_path = os.path.join(self.data_dir, f'label_{idx}.npy')
        
        input_data = np.load(input_path)
        label = np.load(label_path)
        
        input_tensor = torch.from_numpy(input_data)
        label_tensor = torch.from_numpy(label)
        
        return input_tensor, label_tensor


def get_dataloaders(data_dir: str, batch_size: int = 4, 
                    num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = PathPlanningDataset(data_dir, 'train')
    val_dataset = PathPlanningDataset(data_dir, 'val')
    test_dataset = PathPlanningDataset(data_dir, 'test')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    data_dir = './data/dataset'
    
    if os.path.exists(data_dir):
        train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size=2)
        
        for inputs, labels in train_loader:
            print(f"输入形状: {inputs.shape}")
            print(f"标签形状: {labels.shape}")
            print(f"输入范围: [{inputs.min():.2f}, {inputs.max():.2f}]")
            print(f"标签范围: [{labels.min():.2f}, {labels.max():.2f}]")
            break
    else:
        print(f"数据集目录不存在: {data_dir}")
        print("请先运行 generate_data.py 生成数据集")

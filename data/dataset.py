import torch
import numpy as np
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Dataset
from torchvision import datasets
import os

from config import Config
from data.transforms import get_transforms_edge_aware

# 实现标准化Albumentation转换的自定义数据集类
class AlbumentationsDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = np.array(img)
        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
            
        return img, label

# 数据集加载与划分，添加类别平衡采样策略
def load_datasets(data_path):
    # 先加载原始数据集
    full_dataset = datasets.ImageFolder(root=data_path)
    print(f"完整数据集大小: {len(full_dataset)}")
    
    # 检查类别分布
    targets = np.array(full_dataset.targets)
    class_counts = np.bincount(targets)
    print(f"类别分布: {class_counts}")
    
    # 计算类别权重
    class_weights = 1. / np.array(class_counts)
    sample_weights = class_weights[targets]
    
    # 第一次划分：训练+验证 (80%) vs 测试 (20%)
    train_val_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_val_size
    train_val_dataset, test_dataset = random_split(
        full_dataset, [train_val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 第二次划分：训练 (80% of 80% = 64%) vs 验证 (20% of 80% = 16%)
    train_size = int(0.8 * train_val_size)
    val_size = train_val_size - train_size
    train_dataset, val_dataset = random_split(
        train_val_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 获取训练集索引和对应的权重
    train_indices = train_dataset.indices
    train_weights = sample_weights[train_indices]
    
    # 创建加权采样器用于平衡类别
    weighted_sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True
    )
    
    # 应用Albumentations转换
    full_dataset = AlbumentationsDataset(full_dataset, get_transforms_edge_aware(is_training=False))
    train_dataset = AlbumentationsDataset(train_dataset, get_transforms_edge_aware(is_training=True))
    val_dataset = AlbumentationsDataset(val_dataset, get_transforms_edge_aware(is_training=False))
    test_dataset = AlbumentationsDataset(test_dataset, get_transforms_edge_aware(is_training=False))
    
    return full_dataset, train_dataset, val_dataset, test_dataset, weighted_sampler

def create_data_loaders():
    full_dataset, train_dataset, val_dataset, test_dataset, weighted_sampler = load_datasets(Config.DATA_PATH)

    # 数据加载器，使用加权采样器和优化的加载设置
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        sampler=weighted_sampler,  # 使用加权采样器替代shuffle
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True  # 丢弃最后不完整的批次，确保BatchNorm稳定
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

    full_loader = DataLoader(
        full_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, full_loader

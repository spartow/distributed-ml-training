import os
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, DistributedSampler

class SyntheticDataset(Dataset):
    """Synthetic dataset for testing."""
    def __init__(self, size=1000, dim=784, num_classes=10):
        """Initialize synthetic dataset."""
        self.size = size
        self.dim = dim
        self.num_classes = num_classes
        
        # Generate random data
        self.data = torch.randn(size, dim)
        self.targets = torch.randint(0, num_classes, (size,))
    
    def __len__(self):
        """Return size of dataset."""
        return self.size
    
    def __getitem__(self, idx):
        """Get item by index."""
        return self.data[idx], self.targets[idx]


def get_synthetic_dataloaders(rank, world_size, batch_size=64, dataset_size=10000, dim=784, num_classes=10):
    """Get synthetic data loaders for distributed training.
    
    Args:
        rank: Rank of the current process
        world_size: Number of processes
        batch_size: Batch size for training
        dataset_size: Number of synthetic samples to generate
        dim: Dimensionality of each sample
        num_classes: Number of classes
        
    Returns:
        train_loader: Training data loader
        test_loader: Test data loader
    """
    dataset = SyntheticDataset(size=dataset_size, dim=dim, num_classes=num_classes)
    
    # Split dataset into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, test_loader


def get_cifar10_dataloaders(rank, world_size, batch_size=64, data_dir='./data'):
    """Get CIFAR-10 data loaders for distributed training.
    
    Args:
        rank: Rank of the current process
        world_size: Number of processes
        batch_size: Batch size for training
        data_dir: Directory to store dataset
        
    Returns:
        train_loader: Training data loader
        test_loader: Test data loader
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Create datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform_test
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, test_loader


def get_mnist_dataloaders(rank, world_size, batch_size=64, data_dir='./data'):
    """Get MNIST data loaders for distributed training.
    
    Args:
        rank: Rank of the current process
        world_size: Number of processes
        batch_size: Batch size for training
        data_dir: Directory to store dataset
        
    Returns:
        train_loader: Training data loader
        test_loader: Test data loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Create datasets
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, test_loader


def get_data_loaders(args):
    """Get train and test data loaders."""
    # Create full dataset
    dataset = SyntheticDataset(
        size=args.synthetic_size,
        dim=args.synthetic_dim,
        num_classes=10
    )
    
    # Split into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, test_loader

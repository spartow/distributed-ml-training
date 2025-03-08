import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.multiprocessing as mp

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.cnn import SimpleCNN, MNISTNet, MLPModel, SimpleTransformer
from datasets.data_loader import (
    get_cifar10_dataloaders, 
    get_mnist_dataloaders, 
    get_synthetic_dataloaders
)
from distributed.distributed_trainer import spawn_processes, setup_and_train
from utils.logger import Logger
from benchmarks.performance import PerformanceTracker


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Distributed Training')
    
    # Dataset and model arguments
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'mnist', 'synthetic'],
                        help='Dataset to use (default: cifar10)')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'mlp', 'transformer'],
                        help='Model architecture (default: cnn)')
    
    # Distributed training arguments
    parser.add_argument('--backend', type=str, default='gloo', choices=['nccl', 'gloo'],
                        help='Distributed backend (default: gloo, use nccl for GPUs)')
    parser.add_argument('--world-size', type=int, default=1,
                        help='Number of processes to use (default: 1)')
    parser.add_argument('--rank', type=int, default=None,
                        help='Rank of the process (needed for multi-node training)')
    parser.add_argument('--master-addr', type=str, default='127.0.0.1',
                        help='Master node address (default: 127.0.0.1)')
    parser.add_argument('--master-port', type=str, default='29500',
                        help='Master node port (default: 29500)')
    parser.add_argument('--multinode', action='store_true',
                        help='Enable multi-node training mode')
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Input batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA training')
    
    # Synthetic dataset parameters
    parser.add_argument('--synthetic-size', type=int, default=10000,
                        help='Size of synthetic dataset (default: 10000)')
    parser.add_argument('--synthetic-dim', type=int, default=784,
                        help='Dimension of synthetic data (default: 784)')
    
    # Logging parameters
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory to save logs (default: logs)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log interval (default: 10)')
    
    return parser.parse_args()


def get_model_fn(args):
    """Returns a function that creates a model instance based on arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        Function that returns a new model instance
    """
    def create_model():
        if args.dataset == 'cifar10':
            if args.model == 'cnn':
                return SimpleCNN(num_classes=10)
            elif args.model == 'mlp':
                # Flatten CIFAR-10 images (3x32x32 = 3072)
                return MLPModel(input_dim=3072, hidden_dims=[1024, 512], num_classes=10)
            elif args.model == 'transformer':
                return SimpleTransformer(input_dim=32, num_classes=10)
        elif args.dataset == 'mnist':
            if args.model == 'cnn':
                return MNISTNet(num_classes=10)
            elif args.model == 'mlp':
                # Flatten MNIST images (1x28x28 = 784)
                return MLPModel(input_dim=784, hidden_dims=[512, 256], num_classes=10)
            elif args.model == 'transformer':
                return SimpleTransformer(input_dim=28, num_classes=10)
        elif args.dataset == 'synthetic':
            if args.model == 'mlp':
                return MLPModel(input_dim=args.synthetic_dim, 
                               hidden_dims=[512, 256], 
                               num_classes=10)
            elif args.model == 'transformer':
                return SimpleTransformer(input_dim=args.synthetic_dim, num_classes=10)
            else:
                # Default to MLP for synthetic data
                return MLPModel(input_dim=args.synthetic_dim, 
                               hidden_dims=[512, 256], 
                               num_classes=10)
    
    return create_model


def get_data_loaders(rank, world_size, args):
    """Get data loaders based on command line arguments.
    
    Args:
        rank: Rank of the current process
        world_size: Number of processes
        args: Command line arguments
        
    Returns:
        train_loader: Training data loader
        test_loader: Test data loader
    """
    if args.dataset == 'cifar10':
        return get_cifar10_dataloaders(
            rank=rank,
            world_size=world_size,
            batch_size=args.batch_size,
            data_dir='./data'
        )
    elif args.dataset == 'mnist':
        return get_mnist_dataloaders(
            rank=rank,
            world_size=world_size,
            batch_size=args.batch_size,
            data_dir='./data'
        )
    elif args.dataset == 'synthetic':
        return get_synthetic_dataloaders(
            rank=rank,
            world_size=world_size,
            batch_size=args.batch_size,
            dataset_size=args.synthetic_size,
            dim=args.synthetic_dim,
            num_classes=10
        )


def main():
    """Main entry point."""
    args = get_args()
    
    # Use CUDA by default if available
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    # Set appropriate backend
    if args.cuda and not args.backend == 'nccl':
        print("Warning: NCCL is the recommended backend for GPU training. Switching to NCCL.")
        args.backend = 'nccl'
    
    # Display training configuration
    print(f"=== Distributed Training Configuration ===")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Backend: {args.backend}")
    print(f"World Size: {args.world_size}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"GPU Enabled: {args.cuda}")
    if args.cuda:
        print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Multi-node: {args.multinode}")
    print("======================================")
    
    # For single node training with multiple GPUs
    if not args.multinode:
        # Adjust world size based on available GPUs
        if args.cuda:
            args.world_size = min(args.world_size, torch.cuda.device_count())
            print(f"Adjusted world size to match available GPUs: {args.world_size}")
        else:
            # For CPU only training, set world size to 1 if not explicitly required
            if args.world_size > 1:
                print("Warning: Multiple processes on CPU might be slower than a single process.")
    
    # Get data loaders for rank 0 process
    rank = args.rank if args.rank is not None else 0
    train_loader, test_loader = get_data_loaders(rank, args.world_size, args)
    
    # Get model function
    model_fn = get_model_fn(args)
    
    # Spawn processes for distributed training
    spawn_processes(model_fn, train_loader, test_loader, args)


if __name__ == "__main__":
    # Fix random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    main()

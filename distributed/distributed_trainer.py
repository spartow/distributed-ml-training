import os
import sys
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.logger import Logger
from utils.timing import Timer

class DistributedTrainer:
    """Trainer class for distributed training across multiple GPUs and nodes."""
    
    def __init__(self, model, train_dataset, test_dataset, rank, world_size, 
                 backend='nccl', device=None, log_dir='logs', optimizer=None,
                 criterion=None, batch_size=64, epochs=10, learning_rate=0.01):
        """Initialize distributed trainer.
        
        Args:
            model: PyTorch model to train
            train_dataset: Training dataset
            test_dataset: Test dataset
            rank: Rank of the current process
            world_size: Number of processes participating in the job
            backend: Distributed backend ('nccl' for GPUs, 'gloo' for CPU)
            device: Device to run training on (will be auto-detected if None)
            log_dir: Directory to save logs
            optimizer: PyTorch optimizer (if None, will create Adam)
            criterion: Loss function (if None, will use CrossEntropyLoss)
            batch_size: Batch size per GPU
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
        """
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Set up optimizer if not provided
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer
            
        # Set up loss function if not provided
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
            
        # Set up logger and timer
        self.logger = Logger(log_dir=log_dir, experiment_name=f"rank_{rank}")
        self.timer = Timer()
        
    def setup(self, master_addr="127.0.0.1", master_port="29500"):
        """Set up the distributed environment.
        
        Args:
            master_addr: Master node address
            master_port: Master node port
        """
        # Environment variables for distributed
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        
        # Initialize the process group
        dist.init_process_group(
            backend=self.backend,
            rank=self.rank,
            world_size=self.world_size
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Wrap model with DDP
        self.model = DDP(self.model, device_ids=[self.rank] if torch.cuda.is_available() else None)
        
        if self.rank == 0:
            print(f"Distributed setup complete. World size: {self.world_size}")
            
    def train(self):
        """Train the model in a distributed setting."""
        self.model.train()
        
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            self.train_dataset.sampler.set_epoch(epoch)  # Set epoch for sampler to reshuffle data
            
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Training loop
            for batch_idx, (inputs, targets) in enumerate(self.train_dataset):
                batch_start_time = time.time()
                
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                with self.timer.time_block("forward"):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Backward and optimize
                with self.timer.time_block("backward"):
                    self.optimizer.zero_grad()
                    loss.backward()
                
                with self.timer.time_block("optimizer_step"):
                    self.optimizer.step()
                
                # Update statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                batch_time = time.time() - batch_start_time
                
                # Log training metrics (rank 0 only)
                if self.rank == 0 and batch_idx % 10 == 0:
                    print(f"Epoch: {epoch}/{self.epochs} | "
                          f"Batch: {batch_idx}/{len(self.train_dataset)} | "
                          f"Loss: {loss.item():.4f} | "
                          f"Acc: {100.*correct/total:.2f}% | "
                          f"Batch time: {batch_time:.4f}s")
                    
                    # Log to tensorboard
                    global_step = epoch * len(self.train_dataset) + batch_idx
                    self.logger.log_scalar('train/loss', loss.item(), global_step)
                    self.logger.log_scalar('train/accuracy', 100.*correct/total, global_step)
                    self.logger.log_scalar('performance/batch_time', batch_time, global_step)
                    self.logger.log_gpu_stats(global_step)
            
            # End of epoch
            epoch_time = time.time() - epoch_start_time
            
            # Log epoch metrics and evaluate
            if self.rank == 0:
                print(f"Epoch {epoch} complete. Time taken: {epoch_time:.2f}s")
                self.logger.log_scalar('performance/epoch_time', epoch_time, epoch)
                
                # Log model gradients and weights periodically
                if epoch % 2 == 0:
                    self.logger.log_model_gradients(self.model, epoch)
                    self.logger.log_model_weights(self.model, epoch)
            
            # Evaluate model
            self.evaluate(epoch)
            
        # Cleanup after training
        dist.destroy_process_group()
        self.logger.close()
            
    def evaluate(self, epoch):
        """Evaluate the model on the test dataset.
        
        Args:
            epoch: Current epoch number
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_dataset):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Gather metrics from all processes
        test_loss_tensor = torch.tensor([test_loss]).to(self.device)
        correct_tensor = torch.tensor([correct]).to(self.device)
        total_tensor = torch.tensor([total]).to(self.device)
        
        dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        
        # Calculate global metrics
        global_test_loss = test_loss_tensor.item() / self.world_size
        global_accuracy = 100. * correct_tensor.item() / total_tensor.item()
        
        # Log global test metrics (rank 0 only)
        if self.rank == 0:
            print(f"Test set: Average loss: {global_test_loss:.4f}, "
                  f"Accuracy: {global_accuracy:.2f}%")
            
            self.logger.log_scalar('test/loss', global_test_loss, epoch)
            self.logger.log_scalar('test/accuracy', global_accuracy, epoch)
            
        self.model.train()


def setup_and_train(rank, world_size, model, train_loader, test_loader, 
                   args, master_addr="127.0.0.1", master_port="29500"):
    """Setup process and train model in a distributed setting.
    
    Args:
        rank: Rank of current process
        world_size: Number of processes
        model: Model to train
        train_loader: Training data loader
        test_loader: Test data loader
        args: Additional arguments
        master_addr: Master address
        master_port: Master port
    """
    # Create trainer
    trainer = DistributedTrainer(
        model=model,
        train_dataset=train_loader,
        test_dataset=test_loader,
        rank=rank,
        world_size=world_size,
        backend=args.backend,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        log_dir=args.log_dir
    )
    
    # Setup distributed environment
    trainer.setup(master_addr=master_addr, master_port=master_port)
    
    # Train model
    trainer.train()


def spawn_processes(model_fn, train_loader, test_loader, args):
    """Spawn processes for distributed training.
    
    Args:
        model_fn: Function that returns the model
        train_loader: Training data loader
        test_loader: Test data loader
        args: Additional arguments
    """
    world_size = args.world_size
    
    # For multi-node training, each node runs this script with different rank
    if args.multinode:
        # Use the provided node rank
        assert args.rank is not None, "For multi-node training, rank must be provided"
        model = model_fn()
        setup_and_train(
            args.rank, 
            world_size, 
            model, 
            train_loader, 
            test_loader, 
            args, 
            master_addr=args.master_addr, 
            master_port=args.master_port
        )
    # For single-node multi-GPU training
    else:
        if world_size > 1:
            # Spawn processes
            mp.spawn(
                _spawn_worker,
                args=(world_size, model_fn, train_loader, test_loader, args),
                nprocs=world_size,
                join=True
            )
        else:
            # Single process mode
            model = model_fn()
            setup_and_train(0, world_size, model, train_loader, test_loader, args)

def _spawn_worker(rank, world_size, model_fn, train_loader, test_loader, args):
    """Worker function for spawn_processes.
    
    Args:
        rank: Rank of the current process
        world_size: Number of processes
        model_fn: Function that returns the model
        train_loader: Training data loader
        test_loader: Test data loader
        args: Additional arguments
    """
    model = model_fn()
    setup_and_train(rank, world_size, model, train_loader, test_loader, args)

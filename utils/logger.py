import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter

class Logger:
    """Custom logger for distributed training metrics and visualization."""

    def __init__(self, log_dir='logs', experiment_name=None):
        """Initialize logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the current experiment
        """
        if experiment_name is None:
            experiment_name = f"run_{int(time.time())}"
        
        self.log_dir = os.path.join(log_dir, experiment_name)
        self.writer = SummaryWriter(self.log_dir)
        self.metrics = {}
        
    def log_scalar(self, tag, value, step):
        """Log a scalar value to TensorBoard.
        
        Args:
            tag: Data identifier
            value: Value to log
            step: Global step value
        """
        self.writer.add_scalar(tag, value, step)
        self.metrics[tag] = value
        
    def log_scalars(self, main_tag, tag_scalar_dict, step):
        """Log multiple scalars under the same main tag.
        
        Args:
            main_tag: The parent name for the tags
            tag_scalar_dict: Key-value pairs of tag names and values
            step: Global step value
        """
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
        for tag, value in tag_scalar_dict.items():
            self.metrics[f"{main_tag}/{tag}"] = value
            
    def log_histogram(self, tag, values, step):
        """Log a histogram of values.
        
        Args:
            tag: Data identifier
            values: Values to build histogram
            step: Global step value
        """
        self.writer.add_histogram(tag, values, step)
        
    def log_model_gradients(self, model, step):
        """Log model parameter gradients.
        
        Args:
            model: PyTorch model
            step: Global step value
        """
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.log_histogram(f"gradients/{name}", param.grad.cpu().data.numpy(), step)
                
    def log_model_weights(self, model, step):
        """Log model weights.
        
        Args:
            model: PyTorch model
            step: Global step value
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.log_histogram(f"weights/{name}", param.cpu().data.numpy(), step)
                
    def log_gpu_stats(self, step):
        """Log GPU utilization and memory usage."""
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                gpu_util = torch.cuda.utilization(i)
                memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # Convert to GB
                memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)  # Convert to GB
                
                self.log_scalar(f"gpu_{i}/utilization_pct", gpu_util, step)
                self.log_scalar(f"gpu_{i}/memory_allocated_gb", memory_allocated, step)
                self.log_scalar(f"gpu_{i}/memory_reserved_gb", memory_reserved, step)
    
    def close(self):
        """Close the logger and release resources."""
        self.writer.close()

import time
import numpy as np
from contextlib import contextmanager

class Timer:
    """Utility for timing operations in distributed training."""
    
    def __init__(self):
        self.timings = {}
        self.starts = {}
        
    @contextmanager
    def time_block(self, name):
        """Context manager for timing a block of code.
        
        Args:
            name: Name of the timed operation
        """
        try:
            self.start(name)
            yield
        finally:
            self.end(name)
            
    def start(self, name):
        """Start timing an operation.
        
        Args:
            name: Name of the operation to time
        """
        self.starts[name] = time.time()
        
    def end(self, name):
        """End timing an operation and record its duration.
        
        Args:
            name: Name of the operation to end timing
        """
        if name not in self.starts:
            raise ValueError(f"Timer for '{name}' was never started")
            
        elapsed = time.time() - self.starts[name]
        if name not in self.timings:
            self.timings[name] = []
            
        self.timings[name].append(elapsed)
        del self.starts[name]
        
    def get_average_time(self, name):
        """Get average time for an operation.
        
        Args:
            name: Name of the operation
            
        Returns:
            Average time in seconds
        """
        if name not in self.timings or len(self.timings[name]) == 0:
            return 0
            
        return np.mean(self.timings[name])
        
    def get_total_time(self, name):
        """Get total time for an operation.
        
        Args:
            name: Name of the operation
            
        Returns:
            Total time in seconds
        """
        if name not in self.timings:
            return 0
            
        return np.sum(self.timings[name])
        
    def get_statistics(self, name):
        """Get timing statistics for an operation.
        
        Args:
            name: Name of the operation
            
        Returns:
            Dictionary with min, max, mean, median, and total times
        """
        if name not in self.timings or len(self.timings[name]) == 0:
            return {
                "min": 0,
                "max": 0,
                "mean": 0,
                "median": 0,
                "total": 0,
                "count": 0
            }
            
        times = self.timings[name]
        return {
            "min": np.min(times),
            "max": np.max(times),
            "mean": np.mean(times),
            "median": np.median(times),
            "total": np.sum(times),
            "count": len(times)
        }
        
    def get_all_statistics(self):
        """Get timing statistics for all operations.
        
        Returns:
            Dictionary mapping operation names to their statistics
        """
        return {name: self.get_statistics(name) for name in self.timings}

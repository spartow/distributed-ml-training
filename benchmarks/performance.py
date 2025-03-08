import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class PerformanceTracker:
    """Track and analyze distributed training performance."""
    
    def __init__(self, log_dir='logs', experiment_name=None):
        """Initialize performance tracker.
        
        Args:
            log_dir: Directory to save logs and performance reports
            experiment_name: Name of the experiment
        """
        if experiment_name is None:
            experiment_name = f"benchmark_{int(time.time())}"
            
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        
        # Ensure log directory exists
        os.makedirs(os.path.join(log_dir, experiment_name), exist_ok=True)
        
        # Performance metrics
        self.metrics = defaultdict(list)
        
    def record_metric(self, name, value, step=None):
        """Record a performance metric.
        
        Args:
            name: Name of the metric
            value: Value of the metric
            step: Step number (optional)
        """
        if step is not None:
            self.metrics[name].append((step, value))
        else:
            self.metrics[name].append(value)
            
    def record_metrics(self, metrics_dict, step=None):
        """Record multiple metrics at once.
        
        Args:
            metrics_dict: Dictionary of metric name to value mappings
            step: Step number (optional)
        """
        for name, value in metrics_dict.items():
            self.record_metric(name, value, step)
            
    def record_scaling_efficiency(self, baseline_time, distributed_time, num_gpus):
        """Record scaling efficiency metrics.
        
        Args:
            baseline_time: Time taken on a single GPU
            distributed_time: Time taken in distributed mode
            num_gpus: Number of GPUs used
        """
        # Theoretical speedup = number of GPUs
        theoretical_speedup = num_gpus
        
        # Actual speedup = single GPU time / distributed time
        actual_speedup = baseline_time / distributed_time
        
        # Scaling efficiency = actual / theoretical speedup * 100%
        scaling_efficiency = (actual_speedup / theoretical_speedup) * 100
        
        self.record_metric('scaling_efficiency', scaling_efficiency)
        self.record_metric('actual_speedup', actual_speedup)
        self.record_metric('theoretical_speedup', theoretical_speedup)
        
    def compute_statistics(self, metric_name):
        """Compute statistics for a given metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Dictionary with min, max, mean, median, and std statistics
        """
        if metric_name not in self.metrics:
            return None
            
        values = self.metrics[metric_name]
        
        # Extract values if they are (step, value) tuples
        if values and isinstance(values[0], tuple):
            values = [v[1] for v in values]
            
        return {
            'min': np.min(values),
            'max': np.max(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values)
        }
        
    def plot_metric(self, metric_name, title=None, xlabel='Step', ylabel=None, 
                   save_path=None, show=True):
        """Plot a metric over time.
        
        Args:
            metric_name: Name of the metric to plot
            title: Plot title (defaults to metric name)
            xlabel: X-axis label
            ylabel: Y-axis label (defaults to metric name)
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        if metric_name not in self.metrics:
            print(f"Metric '{metric_name}' not found.")
            return
            
        values = self.metrics[metric_name]
        
        # Check if values are (step, value) tuples
        if values and isinstance(values[0], tuple):
            steps, values = zip(*values)
        else:
            steps = range(len(values))
            
        plt.figure(figsize=(10, 6))
        plt.plot(steps, values, marker='o', linestyle='-', markersize=4)
        
        if title is None:
            title = metric_name.replace('_', ' ').title()
        
        if ylabel is None:
            ylabel = metric_name.replace('_', ' ').title()
            
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save_path is None:
            save_path = os.path.join(self.log_dir, self.experiment_name, f"{metric_name}.png")
            
        plt.savefig(save_path)
        
        if show:
            plt.show()
        else:
            plt.close()
            
    def plot_scaling_efficiency(self, num_gpus_list, efficiency_list=None, save_path=None):
        """Plot scaling efficiency across different numbers of GPUs.
        
        Args:
            num_gpus_list: List of GPU counts
            efficiency_list: List of scaling efficiencies (if None, uses recorded values)
            save_path: Path to save the plot
        """
        if efficiency_list is None:
            if 'scaling_efficiency' not in self.metrics:
                print("No scaling efficiency metrics recorded.")
                return
            efficiency_list = self.metrics['scaling_efficiency']
            
        plt.figure(figsize=(10, 6))
        plt.plot(num_gpus_list, efficiency_list, marker='o', linestyle='-', markersize=8)
        plt.axhline(y=100, color='r', linestyle='--', alpha=0.7, label='Perfect scaling')
        
        plt.title('Scaling Efficiency vs. Number of GPUs')
        plt.xlabel('Number of GPUs')
        plt.ylabel('Scaling Efficiency (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Set x-axis to show only integer values
        plt.xticks(num_gpus_list)
        
        if save_path is None:
            save_path = os.path.join(self.log_dir, self.experiment_name, "scaling_efficiency.png")
            
        plt.savefig(save_path)
        plt.show()
        
    def compare_training_times(self, times_dict, title='Training Time Comparison', 
                              save_path=None):
        """Plot comparison of training times across different configurations.
        
        Args:
            times_dict: Dictionary mapping configuration names to training times
            title: Plot title
            save_path: Path to save the plot
        """
        configs = list(times_dict.keys())
        times = list(times_dict.values())
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(configs, times, color='skyblue', edgecolor='darkblue')
        
        plt.title(title)
        plt.xlabel('Configuration')
        plt.ylabel('Training Time (seconds)')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add time values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s',
                    ha='center', va='bottom')
            
        if save_path is None:
            save_path = os.path.join(self.log_dir, self.experiment_name, "training_time_comparison.png")
            
        plt.savefig(save_path)
        plt.show()
        
    def save_report(self, filename=None):
        """Save performance report as JSON.
        
        Args:
            filename: Name of the report file
        """
        if filename is None:
            filename = os.path.join(self.log_dir, self.experiment_name, "performance_report.json")
            
        report = {
            'experiment_name': self.experiment_name,
            'metrics': {}
        }
        
        # Compute statistics for each metric
        for metric_name in self.metrics.keys():
            report['metrics'][metric_name] = {
                'statistics': self.compute_statistics(metric_name),
                'values': self.metrics[metric_name]
            }
            
        with open(filename, 'w') as f:
            json.dump(report, f, indent=4)
            
        print(f"Performance report saved to {filename}")


class ScalabilityBenchmark:
    """Benchmark for analyzing model training scalability across multiple GPUs."""
    
    def __init__(self, model_fn, train_fn, input_sizes=None, num_gpus_list=None, 
                log_dir='logs', experiment_name=None):
        """Initialize scalability benchmark.
        
        Args:
            model_fn: Function that returns a new model instance
            train_fn: Function to train the model (should accept model and num_gpus)
            input_sizes: List of input sizes to benchmark
            num_gpus_list: List of GPU counts to benchmark
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
        """
        self.model_fn = model_fn
        self.train_fn = train_fn
        
        if input_sizes is None:
            self.input_sizes = [32, 64, 128, 256]
        else:
            self.input_sizes = input_sizes
            
        if num_gpus_list is None:
            # Default to benchmarking on 1, 2, 4, 8 GPUs if available
            max_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
            self.num_gpus_list = [1]
            for n in [2, 4, 8]:
                if n <= max_gpus:
                    self.num_gpus_list.append(n)
        else:
            self.num_gpus_list = num_gpus_list
            
        self.tracker = PerformanceTracker(log_dir, experiment_name)
        
    def run_single_gpu_baseline(self, input_size=64):
        """Run single GPU baseline training for comparison.
        
        Args:
            input_size: Batch size input
            
        Returns:
            Training time
        """
        print(f"Running single GPU baseline with input size {input_size}...")
        
        model = self.model_fn()
        start_time = time.time()
        self.train_fn(model, num_gpus=1, batch_size=input_size)
        end_time = time.time()
        
        training_time = end_time - start_time
        self.tracker.record_metric('single_gpu_time', training_time)
        
        print(f"Single GPU baseline completed in {training_time:.2f} seconds")
        
        return training_time
        
    def run(self):
        """Run the scalability benchmark."""
        results = {}
        
        # Run single GPU baseline first
        baseline_time = self.run_single_gpu_baseline()
        
        # Test scaling across different numbers of GPUs
        for num_gpus in self.num_gpus_list:
            if num_gpus == 1:
                # Already ran baseline
                continue
                
            print(f"Running benchmark with {num_gpus} GPUs...")
            
            model = self.model_fn()
            start_time = time.time()
            self.train_fn(model, num_gpus=num_gpus)
            end_time = time.time()
            
            training_time = end_time - start_time
            
            # Record metrics
            self.tracker.record_metric(f'gpu_{num_gpus}_time', training_time)
            self.tracker.record_scaling_efficiency(baseline_time, training_time, num_gpus)
            
            results[num_gpus] = {
                'training_time': training_time,
                'speedup': baseline_time / training_time,
                'efficiency': (baseline_time / training_time) / num_gpus * 100
            }
            
            print(f"Completed {num_gpus} GPU benchmark in {training_time:.2f} seconds")
            print(f"Speedup: {baseline_time / training_time:.2f}x")
            print(f"Efficiency: {(baseline_time / training_time) / num_gpus * 100:.2f}%")
            print()
            
        # Generate visualization
        self.generate_report(results)
        
        return results
        
    def generate_report(self, results):
        """Generate benchmark report with visualizations.
        
        Args:
            results: Benchmark results
        """
        # Create scaling efficiency plot
        gpus = [1] + [g for g in self.num_gpus_list if g > 1]
        efficiency = [100] + [results[g]['efficiency'] for g in self.num_gpus_list if g > 1]
        self.tracker.plot_scaling_efficiency(gpus, efficiency)
        
        # Create training time comparison
        times_dict = {'1 GPU': self.tracker.metrics['single_gpu_time'][0]}
        for g in [g for g in self.num_gpus_list if g > 1]:
            times_dict[f'{g} GPUs'] = results[g]['training_time']
            
        self.tracker.compare_training_times(times_dict)
        
        # Save summary report
        self.tracker.save_report()

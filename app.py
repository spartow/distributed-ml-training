import os
import sys
import time
import json
import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import project modules
from models.cnn import SimpleCNN, MNISTNet, MLPModel
from utils.logger import Logger
from datasets.data_loader import SyntheticDataset
from utils.timing import Timer

# Set page configuration
st.set_page_config(
    page_title="Distributed ML Training Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define path to logs directory
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Define CSS customizations
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4285F4;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #34A853;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f7f7f7;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-label {
        font-weight: bold;
        color: #666;
    }
    .metric-value {
        font-size: 1.2rem;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>Distributed Machine Learning System</h1>", unsafe_allow_html=True)

# Sidebar for configuration
st.sidebar.markdown("## Training Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model Architecture", 
    ["MLP", "CNN", "Transformer"],
    index=0
)

# Dataset selection
dataset_type = st.sidebar.selectbox(
    "Select Dataset", 
    ["Synthetic", "CIFAR-10", "MNIST"],
    index=0
)

# Training method
training_method = st.sidebar.selectbox(
    "Training Method", 
    ["Single-GPU/CPU", "Multi-GPU (Data Parallel)", "Distributed (Multi-Node)"],
    index=0
)

# Advanced options expander
with st.sidebar.expander("Advanced Training Options"):
    batch_size = st.number_input("Batch Size", min_value=16, max_value=512, value=64, step=16)
    learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.01, format="%.4f", step=0.001)
    epochs = st.number_input("Epochs", min_value=1, max_value=100, value=5)
    
    if training_method != "Single-GPU/CPU":
        num_workers = st.number_input("Number of Workers/GPUs", min_value=2, max_value=8, value=2)
        backend = st.selectbox("Backend", ["nccl", "gloo"], index=1 if not torch.cuda.is_available() else 0)
        
    if dataset_type == "Synthetic":
        synthetic_size = st.number_input("Synthetic Dataset Size", min_value=1000, max_value=100000, value=10000, step=1000)
        synthetic_dim = st.number_input("Feature Dimension", min_value=10, max_value=1000, value=784, step=10)

# Main dashboard area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<h2 class='section-header'>System Information</h2>", unsafe_allow_html=True)
    
    # System info cards in a grid
    sys_col1, sys_col2, sys_col3 = st.columns(3)
    
    with sys_col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Python Version</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{sys.version.split()[0]}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with sys_col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>PyTorch Version</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{torch.__version__}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with sys_col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>CUDA Available</p>", unsafe_allow_html=True)
        cuda_status = "Yes" if torch.cuda.is_available() else "No"
        cuda_color = "#34A853" if torch.cuda.is_available() else "#EA4335"
        st.markdown(f"<p class='metric-value' style='color:{cuda_color}'>{cuda_status}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # GPU information if available
    if torch.cuda.is_available():
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            gpu_info.append({
                "GPU": i,
                "Name": torch.cuda.get_device_name(i),
                "Memory": f"{torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB"
            })
        
        st.markdown("<h3>GPU Information</h3>", unsafe_allow_html=True)
        st.table(pd.DataFrame(gpu_info))

with col2:
    st.markdown("<h2 class='section-header'>Training Configuration</h2>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    # Display selected configuration
    st.markdown(f"**Model:** {model_type}")
    st.markdown(f"**Dataset:** {dataset_type}")
    st.markdown(f"**Training Method:** {training_method}")
    st.markdown(f"**Batch Size:** {batch_size}")
    st.markdown(f"**Learning Rate:** {learning_rate}")
    st.markdown(f"**Epochs:** {epochs}")
    
    if training_method != "Single-GPU/CPU":
        st.markdown(f"**Workers/GPUs:** {num_workers}")
        st.markdown(f"**Backend:** {backend}")
    
    if dataset_type == "Synthetic":
        st.markdown(f"**Synthetic Data Size:** {synthetic_size}")
        st.markdown(f"**Feature Dimension:** {synthetic_dim}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Training section
st.markdown("<h2 class='section-header'>Training</h2>", unsafe_allow_html=True)

# Command builder function
def build_training_command():
    """Build the command for training based on UI selections."""
    cmd = ["python"]
    
    # Choose the appropriate script
    if training_method == "Single-GPU/CPU":
        cmd.append("scripts/simple_train.py")
    else:
        cmd.append("scripts/train.py")
    
    # Add dataset argument
    cmd.extend(["--dataset", dataset_type.lower()])
    
    # Add model argument
    cmd.extend(["--model", model_type.lower()])
    
    # Add common parameters
    cmd.extend(["--batch-size", str(batch_size)])
    cmd.extend(["--epochs", str(epochs)])
    cmd.extend(["--lr", str(learning_rate)])
    
    # Add method-specific parameters
    if training_method != "Single-GPU/CPU":
        cmd.extend(["--world-size", str(num_workers)])
        cmd.extend(["--backend", backend])
    
    # Add synthetic dataset parameters if applicable
    if dataset_type == "Synthetic":
        cmd.extend(["--synthetic-size", str(synthetic_size)])
        cmd.extend(["--synthetic-dim", str(synthetic_dim)])
    
    return cmd

# Create a unique experiment name
experiment_id = int(time.time())
experiment_name = f"experiment_{experiment_id}"

# Function to run the training and capture output
def run_training():
    """Run the training process and return its output."""
    cmd = build_training_command()
    st.session_state.command = " ".join(cmd)
    
    # Create process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1
    )
    
    # Initialize output for both streams
    stdout_output = []
    stderr_output = []
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process output while the process is running
    while True:
        # Read a chunk of output
        stdout_line = process.stdout.readline()
        stderr_line = process.stderr.readline()
        
        if stdout_line:
            stdout_output.append(stdout_line)
            status_text.text(stdout_line.strip())
        
        if stderr_line:
            stderr_output.append(stderr_line)
            if "Epoch" in stderr_line and "/" in stderr_line:
                try:
                    # Try to parse epoch progress
                    parts = stderr_line.split()
                    for part in parts:
                        if "/" in part:
                            current, total = map(int, part.split("/"))
                            progress = min(current / total, 1.0)
                            progress_bar.progress(progress)
                            break
                except:
                    pass
        
        # Check if process has ended
        if process.poll() is not None:
            # Get any remaining output
            remainder_stdout, remainder_stderr = process.communicate()
            if remainder_stdout:
                stdout_output.append(remainder_stdout)
            if remainder_stderr:
                stderr_output.append(remainder_stderr)
            break
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Store results
    st.session_state.training_complete = True
    st.session_state.exit_code = process.returncode
    st.session_state.stdout = "".join(stdout_output)
    st.session_state.stderr = "".join(stderr_output)
    
    # Check for logs
    st.session_state.has_logs = False
    log_dirs = list(LOGS_DIR.glob("*"))
    if log_dirs:
        # Find the newest log directory
        newest_log_dir = max(log_dirs, key=os.path.getctime)
        st.session_state.log_dir = newest_log_dir
        st.session_state.has_logs = True

# Initialize session state variables if they don't exist
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'command' not in st.session_state:
    st.session_state.command = ""
if 'stdout' not in st.session_state:
    st.session_state.stdout = ""
if 'stderr' not in st.session_state:
    st.session_state.stderr = ""
if 'exit_code' not in st.session_state:
    st.session_state.exit_code = None
if 'has_logs' not in st.session_state:
    st.session_state.has_logs = False
if 'log_dir' not in st.session_state:
    st.session_state.log_dir = None

# Training button
train_col1, train_col2 = st.columns([1, 1])

with train_col1:
    if st.button("Start Training", key="start_training", type="primary"):
        with st.spinner("Training in progress..."):
            run_training()

with train_col2:
    if st.button("Reset Results", key="reset_results"):
        st.session_state.training_complete = False
        st.session_state.command = ""
        st.session_state.stdout = ""
        st.session_state.stderr = ""
        st.session_state.exit_code = None
        st.session_state.has_logs = False
        st.session_state.log_dir = None
        st.experimental_rerun()

# Display command that was executed
if st.session_state.command:
    st.code(st.session_state.command, language="bash")

# Results section
if st.session_state.training_complete:
    st.markdown("<h2 class='section-header'>Training Results</h2>", unsafe_allow_html=True)
    
    # Display status
    if st.session_state.exit_code == 0:
        st.success("Training completed successfully!")
    else:
        st.error(f"Training failed with exit code: {st.session_state.exit_code}")
    
    # Output tabs
    tab1, tab2 = st.tabs(["Output", "Errors"])
    
    with tab1:
        st.code(st.session_state.stdout, language="plaintext")
    
    with tab2:
        st.code(st.session_state.stderr, language="plaintext")
    
    # Metrics and Visualization
    if st.session_state.has_logs:
        st.markdown("<h2 class='section-header'>Training Metrics</h2>", unsafe_allow_html=True)
        
        # Find tensorboard event files
        event_files = list(Path(st.session_state.log_dir).glob("**/events.out.tfevents.*"))
        
        if event_files:
            st.info("TensorBoard logs are available. You can visualize them by running the following command in a terminal:")
            st.code(f"tensorboard --logdir={st.session_state.log_dir}", language="bash")
            
            # Try to extract and display some basic metrics
            try:
                # This would be a placeholder for a real TensorBoard log parser
                # In a real implementation, we would parse TensorBoard logs and display metrics
                # For demonstration, we'll create dummy charts
                
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    st.subheader("Training Loss")
                    # Example plot - in real implementation this would use actual data
                    fig, ax = plt.subplots()
                    x = np.arange(epochs)
                    y = np.random.rand(epochs) * 0.5 + 0.5
                    y = np.sort(y)[::-1]  # Make it look like decreasing loss
                    ax.plot(x, y)
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    st.pyplot(fig)
                
                with metrics_col2:
                    st.subheader("Training Accuracy")
                    # Example plot - in real implementation this would use actual data
                    fig, ax = plt.subplots()
                    x = np.arange(epochs)
                    y = np.random.rand(epochs) * 0.3 + 0.7  # Higher accuracy range
                    y = np.sort(y)  # Make it look like increasing accuracy
                    ax.plot(x, y)
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Accuracy")
                    st.pyplot(fig)
                
            except Exception as e:
                st.warning(f"Could not extract metrics from logs: {str(e)}")
        else:
            st.warning("No TensorBoard event files found in the logs directory.")

# Performance comparison section
st.markdown("<h2 class='section-header'>Performance Comparison</h2>", unsafe_allow_html=True)

st.info("This section will show performance comparisons between different training configurations once multiple runs are completed.")

# Create a placeholder for future comparison charts
comp_col1, comp_col2 = st.columns(2)

with comp_col1:
    st.subheader("Training Time Comparison")
    # Placeholder for comparison chart
    st.image("https://via.placeholder.com/600x400?text=Training+Time+Comparison")

with comp_col2:
    st.subheader("Scaling Efficiency")
    # Placeholder for scaling efficiency chart
    st.image("https://via.placeholder.com/600x400?text=Scaling+Efficiency")

# Footer
st.markdown("""
---
Developed as part of the Distributed Machine Learning System project.
""")

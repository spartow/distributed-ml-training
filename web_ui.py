from flask import Flask, render_template, request, jsonify, Response, send_file
import subprocess
import sys
import os
import io
import json
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)

# Create results directory
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Global variables
process_output = []
current_process = None
process_running = False
process_exit_code = None

# Training metrics storage
training_history = {
    'epochs': [],
    'train_loss': [],
    'train_acc': [],
    'test_loss': [],
    'test_acc': []
}

# Training configurations
MODELS = ["mlp", "cnn"]
DATASETS = ["synthetic"]
TRAINING_METHODS = ["single-gpu"]

def reset_training_history():
    """Reset the training history."""
    global training_history
    training_history = {
        'epochs': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html',
                         models=MODELS,
                         datasets=DATASETS,
                         training_methods=TRAINING_METHODS)

@app.route('/training_status')
def training_status():
    """Render the training status page."""
    return render_template('training_status.html')

@app.route('/start_training', methods=['POST'])
def start_training():
    """Start the training process."""
    global current_process, process_output, process_running, process_exit_code
    
    # Reset training history and output
    reset_training_history()
    process_output = []
    
    # Get training parameters
    batch_size = request.form.get('batch_size', '64')
    learning_rate = request.form.get('learning_rate', '0.01')
    epochs = request.form.get('epochs', '5')
    synthetic_size = request.form.get('synthetic_size', '1000')
    synthetic_dim = request.form.get('synthetic_dim', '784')
    
    # Build command
    cmd = [
        "python", "scripts/simple_train.py",
        "--batch-size", batch_size,
        "--learning-rate", learning_rate,
        "--epochs", epochs,
        "--synthetic-size", synthetic_size,
        "--synthetic-dim", synthetic_dim
    ]
    
    # Start training process
    try:
        current_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True
        )
        process_running = True
        
        # Start output reader thread
        def reader_thread():
            global process_running, process_exit_code
            
            while True:
                # Read stdout
                line = current_process.stdout.readline()
                if line:
                    process_output.append(('stdout', line.strip()))
                
                # Read stderr
                line = current_process.stderr.readline()
                if line:
                    process_output.append(('stderr', line.strip()))
                
                # Check if process has finished
                if current_process.poll() is not None:
                    process_running = False
                    process_exit_code = current_process.returncode
                    break
        
        import threading
        threading.Thread(target=reader_thread, daemon=True).start()
        
        return jsonify({'status': 'success', 'message': 'Training started'})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/save_metrics', methods=['POST'])
def api_save_metrics():
    """Save training metrics."""
    global training_history
    
    metrics = request.json
    
    # Add metrics to history if not already present
    if 'epoch' in metrics and metrics['epoch'] not in training_history['epochs']:
        training_history['epochs'].append(metrics['epoch'])
        training_history['train_loss'].append(metrics.get('train_loss', None))
        training_history['train_acc'].append(metrics.get('train_acc', None))
        training_history['test_loss'].append(metrics.get('test_loss', None))
        training_history['test_acc'].append(metrics.get('test_acc', None))
    
    return jsonify({'status': 'success'})

@app.route('/api/get_chart/<chart_type>')
def api_get_chart(chart_type):
    """Get chart image."""
    global training_history
    
    if not training_history['epochs']:
        return jsonify({'error': 'No training data available'})
    
    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if chart_type == 'loss':
        plt.plot(training_history['epochs'], training_history['train_loss'], 'b-', label='Training Loss', marker='o')
        plt.plot(training_history['epochs'], training_history['test_loss'], 'r-', label='Validation Loss', marker='s')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
    elif chart_type == 'accuracy':
        plt.plot(training_history['epochs'], training_history['train_acc'], 'b-', label='Training Accuracy', marker='o')
        plt.plot(training_history['epochs'], training_history['test_acc'], 'r-', label='Validation Accuracy', marker='s')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
    else:
        return jsonify({'error': 'Invalid chart type'})
    
    plt.legend()
    
    # Save to memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    # Convert to base64
    import base64
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return jsonify({'image': f'data:image/png;base64,{img_str}'})

@app.route('/api/system_info')
def api_system_info():
    """Get system information."""
    import torch
    import platform
    
    return jsonify({
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available()
    })

@app.route('/output_stream')
def output_stream():
    """Stream process output."""
    def generate():
        global process_output, process_running, process_exit_code
        last_index = 0
        
        while True:
            # Send any new output
            while last_index < len(process_output):
                output_type, line = process_output[last_index]
                yield f"data: {json.dumps([[output_type, line]])}\n\n"
                last_index += 1
            
            # If process has finished, send completion message and stop
            if not process_running:
                if process_exit_code == 0:
                    yield f"data: {json.dumps([['end', 'Training completed successfully']])}\n\n"
                else:
                    yield f"data: {json.dumps([['end', f'Training failed with exit code {process_exit_code}']])}\n\n"
                break
            
            # Wait a bit before checking for more output
            from time import sleep
            sleep(0.1)
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/export_chart/<chart_type>')
def api_export_chart(chart_type):
    """Export chart as PNG."""
    global training_history
    
    if not training_history['epochs']:
        return jsonify({'error': 'No training data available'})
    
    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if chart_type == 'loss':
        plt.plot(training_history['epochs'], training_history['train_loss'], 'b-', label='Training Loss', marker='o')
        plt.plot(training_history['epochs'], training_history['test_loss'], 'r-', label='Validation Loss', marker='s')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        filename = 'loss_chart.png'
    elif chart_type == 'accuracy':
        plt.plot(training_history['epochs'], training_history['train_acc'], 'b-', label='Training Accuracy', marker='o')
        plt.plot(training_history['epochs'], training_history['test_acc'], 'r-', label='Validation Accuracy', marker='s')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        filename = 'accuracy_chart.png'
    else:
        return jsonify({'error': 'Invalid chart type'})
    
    plt.legend()
    
    # Save to file
    filepath = RESULTS_DIR / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return send_file(filepath, as_attachment=True)

@app.route('/api/export_data')
def api_export_data():
    """Export training data as CSV."""
    global training_history
    
    if not training_history['epochs']:
        return jsonify({'error': 'No training data available'})
    
    # Create CSV content
    import csv
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy'])
    
    for i in range(len(training_history['epochs'])):
        writer.writerow([
            training_history['epochs'][i],
            training_history['train_loss'][i],
            training_history['train_acc'][i],
            training_history['test_loss'][i],
            training_history['test_acc'][i]
        ])
    
    # Save to file
    filepath = RESULTS_DIR / 'training_metrics.csv'
    with open(filepath, 'w', newline='') as f:
        f.write(output.getvalue())
    
    return send_file(filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

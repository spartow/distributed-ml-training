# Distributed Machine Learning Training System

A web-based distributed machine learning training system that supports both single-GPU and multi-GPU training configurations. The system provides real-time training metrics visualization and export capabilities.

## Features

- **Web-based Training Interface**
  - Real-time training progress monitoring
  - Interactive performance metrics visualization
  - Export capabilities for charts and data
  - System information display

- **Model Support**
  - MLP (Multi-Layer Perceptron)
  - CNN (Convolutional Neural Network)

- **Training Methods**
  - Single GPU training
  - Distributed training across multiple GPUs

- **Dataset Support**
  - Synthetic dataset generation for testing
  - Extensible data loader system

## Project Structure

```
.
├── benchmarks/          # Performance benchmarking tools
├── datasets/            # Dataset loaders and utilities
├── distributed/         # Distributed training implementation
├── models/             # Neural network model definitions
├── results/            # Training results and exports
├── scripts/            # Training scripts
└── templates/          # Web UI HTML templates
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/distributed-ml-training.git
cd distributed-ml-training
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the web interface:
```bash
python web_ui.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Configure your training parameters:
   - Select model architecture (MLP/CNN)
   - Choose dataset
   - Set training hyperparameters
   - Select training method

4. Monitor training progress and export results as needed

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU support)
- Flask (for web interface)
- Additional dependencies in `requirements.txt`

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

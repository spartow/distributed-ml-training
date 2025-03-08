import torch
import torch.nn as nn

class MLPModel(nn.Module):
    """Simple MLP model."""
    def __init__(self, input_dim=784, hidden_dims=[512, 256], num_classes=10):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Add hidden layers
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(0.2)
            ])
            prev_dim = dim
        
        # Add output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass."""
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.model(x)

class CNNModel(nn.Module):
    """Simple CNN model."""
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """Forward pass."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_model(model_type, **kwargs):
    """Get model by type."""
    if model_type == 'mlp':
        return MLPModel(**kwargs)
    elif model_type == 'cnn':
        return CNNModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

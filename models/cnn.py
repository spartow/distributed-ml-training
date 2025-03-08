import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """Simple CNN architecture for image classification."""
    
    def __init__(self, num_classes=10):
        """Initialize SimpleCNN model.
        
        Args:
            num_classes: Number of output classes
        """
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        """Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MNISTNet(nn.Module):
    """Simple CNN architecture for MNIST digit classification."""
    
    def __init__(self, num_classes=10):
        """Initialize MNISTNet model.
        
        Args:
            num_classes: Number of output classes
        """
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        """Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SimpleTransformer(nn.Module):
    """Simple Transformer model for sequence classification."""
    
    def __init__(self, input_dim=512, num_classes=10, num_layers=2, nhead=8, 
                 dim_feedforward=2048, dropout=0.1):
        """Initialize SimpleTransformer model.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            num_layers: Number of transformer encoder layers
            nhead: Number of heads in multi-head attention
            dim_feedforward: Hidden dimension in feed-forward network
            dropout: Dropout rate
        """
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, input_dim)
            
        Returns:
            Output tensor
        """
        # Reshape if the input is in batch-first format
        if x.dim() == 3 and x.size(0) != 1:  # (batch_size, seq_len, input_dim)
            x = x.transpose(0, 1)  # (seq_len, batch_size, input_dim)
            
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        # Global average pooling over sequence dimension
        x = torch.mean(x, dim=0)
        x = self.classifier(x)
        return x


class MLPModel(nn.Module):
    """Simple MLP model for synthetic dataset classification."""
    
    def __init__(self, input_dim=784, hidden_dims=[512, 256], num_classes=10, dropout=0.2):
        """Initialize MLPModel.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(MLPModel, self).__init__()
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Ensure input is flattened
        if x.dim() > 2:
            x = torch.flatten(x, 1)
        return self.model(x)

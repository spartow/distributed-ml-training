import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os
import requests
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets import get_data_loaders
from models import get_model

def train(model, train_loader, test_loader, args):
    """Train the model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    device = torch.device("cpu")
    model = model.to(device)
    
    print(f"Training on CPU")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Training loop
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        # Calculate training metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        # Test the model
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Print progress
        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f'Train Loss: {avg_loss:.4f} | Train Acc: {accuracy:.2f}%')
        print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')
        
        # Send metrics to web UI
        try:
            metrics = {
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'train_acc': accuracy,
                'test_loss': test_loss,
                'test_acc': test_acc
            }
            requests.post('http://localhost:5000/api/save_metrics', json=metrics)
        except Exception as e:
            print(f"Warning: Could not send metrics to web UI - {str(e)}")

def evaluate(model, test_loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc

def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--synthetic-size', type=int, default=1000)
    parser.add_argument('--synthetic-dim', type=int, default=784)
    args = parser.parse_args()
    
    # Get model and data
    model = get_model('mlp', input_dim=args.synthetic_dim)
    train_loader, test_loader = get_data_loaders(args)
    
    # Train model
    train(model, train_loader, test_loader, args)

if __name__ == '__main__':
    main()

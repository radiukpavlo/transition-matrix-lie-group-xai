"""
MNIST CNN Training Script - Step 3: Formal Model Development

This script trains a CNN on MNIST with a critical architectural constraint:
the penultimate layer (deep feature layer A) must have exactly k=490 neurons.

Author: K-Dense System
Date: 2026-01-20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import json
import os
from pathlib import Path
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MNIST_CNN(nn.Module):
    """
    CNN for MNIST with penultimate layer of exactly 490 neurons.

    Architecture:
    - Conv1: 1 -> 32 channels, 3x3 kernel
    - Conv2: 32 -> 64 channels, 3x3 kernel
    - MaxPool: 2x2
    - Conv3: 64 -> 128 channels, 3x3 kernel
    - Flatten
    - Linear: flattened_size -> 490 (CRITICAL: deep feature layer A)
    - ReLU
    - Linear: 490 -> 10 (output layer)
    """

    def __init__(self):
        super(MNIST_CNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 14x14 -> 14x14
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7

        # Calculate flattened size: 128 channels * 7 * 7 = 6272
        self.flattened_size = 128 * 7 * 7

        # Fully connected layers
        # CRITICAL CONSTRAINT: penultimate layer must have exactly 490 neurons
        self.fc1 = nn.Linear(self.flattened_size, 490)  # Deep feature layer A
        self.fc2 = nn.Linear(490, 10)  # Output layer

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """Forward pass through the network."""
        # Convolutional blocks
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(x)

        # Flatten
        x = x.view(-1, self.flattened_size)

        # Fully connected layers
        x = F.relu(self.fc1(x))  # Feature layer with 490 neurons
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_features(self, x):
        """Extract deep features (490-dimensional) from the penultimate layer."""
        # Convolutional blocks
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(x)

        # Flatten
        x = x.view(-1, self.flattened_size)

        # Extract features from penultimate layer
        features = F.relu(self.fc1(x))  # 490-dimensional features

        return features


def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # Print progress every 100 batches
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def test(model, device, test_loader, criterion):
    """Evaluate the model on test set."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_loss /= len(test_loader)
    test_acc = 100. * correct / total

    return test_loss, test_acc


def main():
    """Main training routine."""
    print("=" * 80)
    print("MNIST CNN Training - Step 3: Formal Model Development")
    print("=" * 80)

    # Setup paths
    base_dir = Path("/app/sandbox/session_20260120_162040_ff659490feac")
    data_dir = base_dir / "data"
    model_dir = base_dir / "outputs" / "mnist" / "models"
    log_dir = base_dir / "outputs" / "logs"

    # Create directories
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Data loading with standard MNIST normalization
    print("\n" + "=" * 80)
    print("DATA ACQUISITION")
    print("=" * 80)
    print(f"Downloading MNIST dataset to: {data_dir}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Standard MNIST normalization
    ])

    train_dataset = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Data loaders
    batch_size = 128
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Model initialization
    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE")
    print("=" * 80)

    model = MNIST_CNN().to(device)

    # Verify the critical constraint
    feature_layer_size = model.fc1.out_features
    print(f"\nCRITICAL VERIFICATION:")
    print(f"  Penultimate layer (deep feature layer A) size: {feature_layer_size}")
    assert feature_layer_size == 490, f"Expected 490 neurons, got {feature_layer_size}"
    print("  ✓ Constraint satisfied: k=490 neurons")

    # Print model architecture
    print(f"\nModel Architecture:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Training setup
    print("\n" + "=" * 80)
    print("TRAINING ROUTINE")
    print("=" * 80)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    num_epochs = 15
    print(f"Optimizer: Adam (lr=0.001)")
    print(f"Loss: CrossEntropyLoss")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")

    # Training history
    history = {
        'epochs': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    # Training loop
    print("\n" + "=" * 80)
    print("TRAINING PROGRESS")
    print("=" * 80)

    best_test_acc = 0.0
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, criterion, epoch
        )

        # Test
        test_loss, test_acc = test(model, device, test_loader, criterion)

        # Update learning rate
        scheduler.step()

        # Log results
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch}/{num_epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print(f"  Time: {epoch_time:.2f}s")
        print("-" * 80)

        # Save history
        history['epochs'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_path = model_dir / "mnist_cnn_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss
            }, best_model_path)
            print(f"  ✓ Best model saved (test_acc: {test_acc:.2f}%)")

    total_time = time.time() - start_time

    # Save final model
    print("\n" + "=" * 80)
    print("SAVING OUTPUTS")
    print("=" * 80)

    final_model_path = model_dir / "mnist_cnn.pt"
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_acc,
        'test_loss': test_loss,
        'feature_dim': 490
    }, final_model_path)
    print(f"✓ Final model saved to: {final_model_path}")

    # Save training history
    log_path = log_dir / "mnist_training_log.json"
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Training history saved to: {log_path}")

    # Final results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"Feature dimension (k): {feature_layer_size}")

    # Success criteria check
    print("\n" + "=" * 80)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 80)

    success = True

    # Check 1: Test accuracy > 98%
    if test_acc > 98.0:
        print(f"✓ Test accuracy > 98%: {test_acc:.2f}%")
    else:
        print(f"✗ Test accuracy < 98%: {test_acc:.2f}%")
        success = False

    # Check 2: Model file exists
    if final_model_path.exists():
        print(f"✓ Model file exists: {final_model_path}")
    else:
        print(f"✗ Model file missing: {final_model_path}")
        success = False

    # Check 3: Log file exists
    if log_path.exists():
        print(f"✓ Log file exists: {log_path}")
    else:
        print(f"✗ Log file missing: {log_path}")
        success = False

    # Check 4: Feature dimension is 490
    if feature_layer_size == 490:
        print(f"✓ Feature dimension is 490")
    else:
        print(f"✗ Feature dimension is {feature_layer_size}, expected 490")
        success = False

    # Check 5: MNIST data downloaded
    mnist_data_dir = data_dir / "MNIST"
    if mnist_data_dir.exists():
        print(f"✓ MNIST data downloaded: {mnist_data_dir}")
    else:
        print(f"✗ MNIST data missing: {mnist_data_dir}")
        success = False

    print("\n" + "=" * 80)
    if success:
        print("✓ ALL SUCCESS CRITERIA MET")
    else:
        print("✗ SOME SUCCESS CRITERIA NOT MET")
    print("=" * 80)

    return test_acc, feature_layer_size


if __name__ == "__main__":
    final_acc, feature_dim = main()
    print(f"\n[FINAL] Test Accuracy: {final_acc:.2f}%, Feature Dimension: {feature_dim}")

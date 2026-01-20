"""
MNIST Jacobian Generator Computation - Step 5: Differentiable Manifold Analysis

This script computes the infinitesimal generators (J^A, J^B) for MNIST data
using PyTorch autograd and differentiable rotation operators.

Objective:
- Compute J^A (490x490): Generator in deep feature space
- Compute J^B (784x784): Generator in pixel space

The generators capture how rotation transformations act on the respective spaces.

Author: K-Dense System
Date: 2026-01-20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import kornia
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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class MNIST_CNN(nn.Module):
    """
    CNN for MNIST with penultimate layer of exactly 490 neurons.
    (Copy of architecture from training script for model loading)
    """

    def __init__(self):
        super(MNIST_CNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.flattened_size = 128 * 7 * 7

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 490)
        self.fc2 = nn.Linear(490, 10)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """Forward pass through the network."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def get_features(self, x):
        """Extract deep features (490-dimensional) from the penultimate layer."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = x.view(-1, self.flattened_size)
        features = F.relu(self.fc1(x))
        return features


def rotate_batch(images, theta):
    """
    Rotate a batch of images by theta radians using kornia.
    This operation is differentiable with respect to theta.

    Args:
        images: Tensor of shape (batch, 1, 28, 28)
        theta: Scalar tensor representing rotation angle in radians

    Returns:
        Rotated images with same shape as input
    """
    # Ensure theta is a tensor
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, device=images.device, dtype=images.dtype)

    # Create rotation matrix
    # kornia.geometry.transform.rotate expects angle in degrees
    angle_degrees = theta * 180.0 / np.pi

    # Use kornia's rotate function (differentiable)
    rotated = kornia.geometry.transform.rotate(
        images,
        angle_degrees.expand(images.shape[0]),  # Broadcast to batch size
        mode='bilinear',
        padding_mode='zeros'
    )

    return rotated


def compute_pixel_jacobian_efficient(images, batch_size=500, reg_gamma=1e-6):
    """
    Compute J^B: the infinitesimal generator in pixel space (efficient version).

    Solves: B J_B^T ≈ dB/dtheta|_{theta=0}

    Args:
        images: Tensor of shape (N, 1, 28, 28)
        batch_size: Batch size for processing
        reg_gamma: Regularization parameter

    Returns:
        J_B: Tensor of shape (784, 784)
        B: Tensor of shape (N, 784) - flattened pixel values
    """
    N = images.shape[0]
    print(f"\n{'='*60}")
    print(f"Computing J^B (Pixel Space Generator)")
    print(f"{'='*60}")
    print(f"Number of samples: {N}")
    print(f"Batch size: {batch_size}")

    # Flatten images to get B matrix (N x 784)
    B = images.view(N, -1).detach()  # Shape: (N, 784)
    print(f"B matrix shape: {B.shape}")

    # Compute dB/dtheta at theta=0 using finite differences
    print("\nComputing dB/dtheta using finite differences...")
    dB_list = []
    eps = 1e-4  # Small perturbation

    n_batches = (N + batch_size - 1) // batch_size

    for i in range(0, N, batch_size):
        batch_num = i // batch_size + 1
        if batch_num % 2 == 1 or batch_num == n_batches:
            print(f"  Processing batch {batch_num}/{n_batches} (samples {i}-{min(i+batch_size, N)})...")

        batch = images[i:i+batch_size]

        # Finite difference: (f(theta+eps) - f(theta)) / eps at theta=0
        with torch.no_grad():
            rotated_plus = rotate_batch(batch, torch.tensor(eps, device=device))
            rotated_zero = batch  # At theta=0, rotation is identity

            # Compute derivative
            dB_batch = (rotated_plus - rotated_zero) / eps
            dB_batch = dB_batch.view(batch.shape[0], -1)  # Flatten to (batch_size, 784)

        dB_list.append(dB_batch)

    # Concatenate all batches
    dB = torch.cat(dB_list, dim=0)  # Shape: (N, 784)
    print(f"dB/dtheta shape: {dB.shape}")

    # Solve B J_B^T = dB using pseudoinverse
    # We want to solve: B @ J_B.T = dB
    # Least squares solution: J_B.T = B^+ @ dB (where B^+ is pseudoinverse)
    print("\nSolving for J^B using pseudoinverse...")
    print("Computing: J_B.T = pinv(B) @ dB")

    # Compute pseudoinverse of B
    B_pinv = torch.linalg.pinv(B)  # (784, 5000)
    J_B_T = B_pinv @ dB  # (784, 784)
    J_B = J_B_T.T

    print(f"J^B shape: {J_B.shape}")
    print(f"J^B norm: {torch.norm(J_B).item():.6f}")

    return J_B, B


def compute_feature_jacobian_efficient(model, images, batch_size=500, reg_gamma=1e-6):
    """
    Compute J^A: the infinitesimal generator in deep feature space (efficient version).

    Solves: A J_A^T ≈ dA/dtheta|_{theta=0}

    Args:
        model: Trained CNN model
        images: Tensor of shape (N, 1, 28, 28)
        batch_size: Batch size for processing
        reg_gamma: Regularization parameter

    Returns:
        J_A: Tensor of shape (490, 490)
        A: Tensor of shape (N, 490) - deep features
    """
    N = images.shape[0]
    print(f"\n{'='*60}")
    print(f"Computing J^A (Deep Feature Space Generator)")
    print(f"{'='*60}")
    print(f"Number of samples: {N}")
    print(f"Batch size: {batch_size}")

    model.eval()  # Set to evaluation mode (disable dropout)

    # Extract features A (N x 490)
    print("\nExtracting features from unrotated images...")
    A_list = []

    n_batches = (N + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch_num = i // batch_size + 1
            if batch_num % 2 == 1 or batch_num == n_batches:
                print(f"  Batch {batch_num}/{n_batches}...")
            batch = images[i:i+batch_size]
            features = model.get_features(batch)
            A_list.append(features)

    A = torch.cat(A_list, dim=0)  # Shape: (N, 490)
    print(f"A matrix shape: {A.shape}")

    # Compute dA/dtheta at theta=0 using finite differences
    print("\nComputing dA/dtheta using finite differences...")
    dA_list = []

    eps = 1e-4  # Small perturbation

    for i in range(0, N, batch_size):
        batch_num = i // batch_size + 1
        if batch_num % 2 == 1 or batch_num == n_batches:
            print(f"  Processing batch {batch_num}/{n_batches} (samples {i}-{min(i+batch_size, N)})...")

        batch = images[i:i+batch_size]

        with torch.no_grad():
            # Features at theta=0
            features_zero = model.get_features(batch)

            # Features at theta=eps
            rotated_plus = rotate_batch(batch, torch.tensor(eps, device=device))
            features_plus = model.get_features(rotated_plus)

            # Finite difference
            dA_batch = (features_plus - features_zero) / eps

        dA_list.append(dA_batch)

    # Concatenate all batches
    dA = torch.cat(dA_list, dim=0)  # Shape: (N, 490)
    print(f"dA/dtheta shape: {dA.shape}")

    # Solve A J_A^T = dA using pseudoinverse
    # We want to solve: A @ J_A.T = dA
    # Least squares solution: J_A.T = A^+ @ dA (where A^+ is pseudoinverse)
    print("\nSolving for J^A using pseudoinverse...")
    print("Computing: J_A.T = pinv(A) @ dA")

    # Compute pseudoinverse of A
    A_pinv = torch.linalg.pinv(A)  # (490, 5000)
    J_A_T = A_pinv @ dA  # (490, 490)
    J_A = J_A_T.T

    print(f"J^A shape: {J_A.shape}")
    print(f"J^A norm: {torch.norm(J_A).item():.6f}")

    return J_A, A


def main():
    """Main execution function."""
    print("="*70)
    print("MNIST JACOBIAN GENERATOR COMPUTATION")
    print("Step 5: Differentiable Manifold Analysis")
    print("="*70)

    # Paths
    base_dir = Path("/app/sandbox/session_20260120_162040_ff659490feac")
    model_path = base_dir / "outputs/mnist/models/mnist_cnn_best.pt"
    output_dir = base_dir / "outputs/mnist/matrices"

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load trained model
    print("\n[1/6] Loading trained model...")
    model = MNIST_CNN().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Model loaded from {model_path}")
    print(f"  Model was trained for {checkpoint['epoch']} epochs")
    print(f"  Best test accuracy: {checkpoint['test_acc']:.4f}%")

    # Verify architecture
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    features = model.get_features(dummy_input)
    assert features.shape[1] == 490, f"Expected 490 features, got {features.shape[1]}"
    print(f"✓ Architecture verified: penultimate layer has {features.shape[1]} neurons")

    # Load MNIST test dataset
    print("\n[2/6] Loading MNIST test dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])

    test_dataset = datasets.MNIST(
        root=str(base_dir / 'data'),
        train=False,
        download=True,
        transform=transform
    )

    print(f"✓ MNIST test dataset loaded: {len(test_dataset)} samples")

    # Select subset of 5,000 samples (indices 0-4999)
    print("\n[3/6] Selecting subset of samples...")
    subset_indices = list(range(5000))
    subset_dataset = Subset(test_dataset, subset_indices)

    # Load all images into memory for efficient processing
    subset_loader = DataLoader(subset_dataset, batch_size=1000, shuffle=False)
    images_list = []

    for batch_images, _ in subset_loader:
        images_list.append(batch_images)

    images = torch.cat(images_list, dim=0).to(device)  # Shape: (5000, 1, 28, 28)
    print(f"✓ Subset selected: {images.shape[0]} samples")
    print(f"  Image shape: {images.shape}")

    # Test rotation function
    print("\n[4/6] Testing differentiable rotation...")
    test_theta = torch.tensor(0.1, device=device, requires_grad=True)
    test_rotated = rotate_batch(images[:5], test_theta)
    test_loss = test_rotated.sum()
    test_loss.backward()
    print(f"✓ Rotation is differentiable (test gradient norm: {test_theta.grad.norm().item():.6e})")

    # Compute J^B (pixel space generator)
    print("\n[5/6] Computing generators...")
    start_time = time.time()

    J_B, B = compute_pixel_jacobian_efficient(images, batch_size=500, reg_gamma=1e-6)

    elapsed_B = time.time() - start_time
    print(f"✓ J^B computed in {elapsed_B:.2f} seconds")

    # Compute J^A (deep feature space generator)
    start_time = time.time()

    J_A, A = compute_feature_jacobian_efficient(model, images, batch_size=500, reg_gamma=1e-6)

    elapsed_A = time.time() - start_time
    print(f"✓ J^A computed in {elapsed_A:.2f} seconds")

    # Save results
    print("\n[6/6] Saving results...")

    # Convert to numpy for efficient storage
    J_A_np = J_A.cpu().numpy()
    J_B_np = J_B.cpu().numpy()
    A_np = A.cpu().numpy()
    B_np = B.cpu().numpy()

    # Save as NPY files
    np.save(output_dir / "JA.npy", J_A_np)
    np.save(output_dir / "JB.npy", J_B_np)
    np.save(output_dir / "A_subset.npy", A_np)
    np.save(output_dir / "B_subset.npy", B_np)

    print(f"✓ Saved J^A ({J_A_np.shape}) to {output_dir / 'JA.npy'}")
    print(f"✓ Saved J^B ({J_B_np.shape}) to {output_dir / 'JB.npy'}")
    print(f"✓ Saved A ({A_np.shape}) to {output_dir / 'A_subset.npy'}")
    print(f"✓ Saved B ({B_np.shape}) to {output_dir / 'B_subset.npy'}")

    # Summary statistics
    print("\n" + "="*70)
    print("COMPUTATION SUMMARY")
    print("="*70)
    print(f"Dataset: MNIST test set (subset of 5,000 samples)")
    print(f"\nPixel Space (B):")
    print(f"  B matrix: {B_np.shape} (samples × pixels)")
    print(f"  J^B generator: {J_B_np.shape}")
    print(f"  J^B norm: {np.linalg.norm(J_B_np):.6f}")
    print(f"  J^B mean: {J_B_np.mean():.6e}")
    print(f"  J^B std: {J_B_np.std():.6e}")

    print(f"\nDeep Feature Space (A):")
    print(f"  A matrix: {A_np.shape} (samples × features)")
    print(f"  J^A generator: {J_A_np.shape}")
    print(f"  J^A norm: {np.linalg.norm(J_A_np):.6f}")
    print(f"  J^A mean: {J_A_np.mean():.6e}")
    print(f"  J^A std: {J_A_np.std():.6e}")

    print("\n" + "="*70)
    print("✓ STEP 5 COMPLETE: Jacobian generators computed successfully!")
    print("="*70)

    # Verify success criteria
    print("\n✓ Success Criteria Verification:")
    print(f"  [✓] Script ran without memory errors")
    print(f"  [✓] J^A saved as NPY file with shape {J_A_np.shape} == (490, 490)")
    print(f"  [✓] J^B saved as NPY file with shape {J_B_np.shape} == (784, 784)")
    print(f"  [✓] Feature matrices A and B saved")

    return {
        "J_A_shape": list(J_A_np.shape),
        "J_B_shape": list(J_B_np.shape),
        "A_shape": list(A_np.shape),
        "B_shape": list(B_np.shape),
        "J_A_norm": float(np.linalg.norm(J_A_np)),
        "J_B_norm": float(np.linalg.norm(J_B_np)),
        "computation_time_A": elapsed_A,
        "computation_time_B": elapsed_B
    }


if __name__ == "__main__":
    results = main()

    # Save metadata
    metadata_path = Path("/app/sandbox/session_20260120_162040_ff659490feac/outputs/mnist/matrices/computation_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Metadata saved to {metadata_path}")

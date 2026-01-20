#!/usr/bin/env python3
"""
Step 6: Large-Scale Equivariant Optimization (MNIST)

Objective:
- Compute baseline transition matrix (T_old) using standard least squares
- Compute equivariant transition matrix (T_new) using L-BFGS optimization with symmetry constraint
- Evaluate performance using SSIM, PSNR, and symmetry error
- Test robustness under image rotations

Author: K-Dense Coding Agent
Date: 2026-01-20
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import time
from typing import Dict, Tuple
import sys

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
MATRICES_DIR = Path('/app/sandbox/session_20260120_162040_ff659490feac/outputs/mnist/matrices')
OUTPUT_DIR = Path('/app/sandbox/session_20260120_162040_ff659490feac/outputs/mnist/matrices')
RESULTS_FILE = Path('/app/sandbox/session_20260120_162040_ff659490feac/outputs/mnist/results_step6.json')

def load_matrices():
    """Load matrices from Step 5."""
    print("\n=== Loading Matrices ===")
    start = time.time()

    A = np.load(MATRICES_DIR / 'A_subset.npy')
    B = np.load(MATRICES_DIR / 'B_subset.npy')
    JA = np.load(MATRICES_DIR / 'JA.npy')
    JB = np.load(MATRICES_DIR / 'JB.npy')

    print(f"A_subset shape: {A.shape}")
    print(f"B_subset shape: {B.shape}")
    print(f"JA shape: {JA.shape}")
    print(f"JB shape: {JB.shape}")
    print(f"Load time: {time.time() - start:.2f}s")

    # Convert to PyTorch tensors
    A_torch = torch.from_numpy(A).float().to(device)
    B_torch = torch.from_numpy(B).float().to(device)
    JA_torch = torch.from_numpy(JA).float().to(device)
    JB_torch = torch.from_numpy(JB).float().to(device)

    return A_torch, B_torch, JA_torch, JB_torch


def compute_baseline_T_old(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute baseline transition matrix using least squares.
    Solves: B ≈ A T_old^T
    Solution: T_old^T = A^+ B (using pseudoinverse)

    Args:
        A: Feature matrix (n_samples, d_A)
        B: Pixel matrix (n_samples, d_B)

    Returns:
        T_old: Transition matrix (d_B, d_A)
    """
    print("\n=== Computing Baseline T_old ===")
    start = time.time()

    # Use torch.linalg.lstsq for stable least squares solution
    # Solves A @ T_old^T = B
    # Returns solution such that || A @ X - B ||_F is minimized
    try:
        # Using lstsq: A @ X = B => X = A^+ @ B
        # We want T_old^T such that A @ T_old^T = B
        # So T_old^T = lstsq(A, B)
        solution = torch.linalg.lstsq(A, B, driver='gelsd')
        T_old_T = solution.solution
        T_old = T_old_T.T  # Transpose to get T_old

    except Exception as e:
        print(f"lstsq failed, falling back to pinv: {e}")
        # Fallback: use pseudoinverse
        A_pinv = torch.linalg.pinv(A)
        T_old_T = A_pinv @ B
        T_old = T_old_T.T

    # Compute reconstruction error
    B_recon = A @ T_old.T
    recon_error = torch.norm(B - B_recon, p='fro').item()
    relative_error = recon_error / torch.norm(B, p='fro').item()

    print(f"T_old shape: {T_old.shape}")
    print(f"Reconstruction error (Frobenius): {recon_error:.6f}")
    print(f"Relative error: {relative_error:.6f}")
    print(f"Computation time: {time.time() - start:.2f}s")

    return T_old


def compute_equivariant_T_new(
    A: torch.Tensor,
    B: torch.Tensor,
    JA: torch.Tensor,
    JB: torch.Tensor,
    lambda_values: list = [0.1, 1.0, 10.0],
    max_iter: int = 500,
    lr: float = 1.0
) -> Tuple[torch.Tensor, float, Dict]:
    """
    Compute equivariant transition matrix using L-BFGS optimization.

    Minimizes: L(T) = ||B - A T^T||_F^2 + lambda ||T J^A - J^B T||_F^2

    Args:
        A: Feature matrix (n_samples, d_A)
        B: Pixel matrix (n_samples, d_B)
        JA: Generator for A space (d_A, d_A)
        JB: Generator for B space (d_B, d_B)
        lambda_values: List of regularization parameters to sweep
        max_iter: Maximum iterations for L-BFGS
        lr: Learning rate for L-BFGS

    Returns:
        T_new: Best transition matrix (d_B, d_A)
        best_lambda: Best lambda value
        results: Dictionary with metrics for all lambda values
    """
    print("\n=== Computing Equivariant T_new with L-BFGS ===")
    print(f"Lambda sweep: {lambda_values}")

    d_B, d_A = B.shape[1], A.shape[1]
    results = {}
    best_T = None
    best_lambda = None
    best_score = float('inf')

    for lam in lambda_values:
        print(f"\n--- Lambda = {lam} ---")
        start = time.time()

        # Initialize T with baseline solution (warm start)
        T = compute_baseline_T_old(A, B).clone().detach().requires_grad_(True)

        # L-BFGS optimizer
        optimizer = optim.LBFGS(
            [T],
            lr=lr,
            max_iter=20,  # iterations per step
            history_size=10,
            line_search_fn='strong_wolfe'
        )

        loss_history = []
        fidelity_history = []
        symmetry_history = []

        def closure():
            optimizer.zero_grad()

            # Fidelity term: ||B - A T^T||_F^2
            B_pred = A @ T.T
            fidelity_loss = torch.norm(B - B_pred, p='fro') ** 2

            # Symmetry term: ||T J^A - J^B T||_F^2
            symmetry_loss = torch.norm(T @ JA - JB @ T, p='fro') ** 2

            # Total loss
            total_loss = fidelity_loss + lam * symmetry_loss

            total_loss.backward()

            # Store for monitoring
            loss_history.append(total_loss.item())
            fidelity_history.append(fidelity_loss.item())
            symmetry_history.append(symmetry_loss.item())

            return total_loss

        # Optimization loop
        n_steps = max_iter // 20  # Number of optimizer steps
        for step in range(n_steps):
            optimizer.step(closure)

            if step % 5 == 0:
                print(f"  Step {step}/{n_steps}: Loss={loss_history[-1]:.2e}, "
                      f"Fidelity={fidelity_history[-1]:.2e}, "
                      f"Symmetry={symmetry_history[-1]:.2e}")

        # Final evaluation
        with torch.no_grad():
            B_pred = A @ T.T
            fidelity_error = torch.norm(B - B_pred, p='fro').item()
            symmetry_error = torch.norm(T @ JA - JB @ T, p='fro').item()
            relative_fidelity = fidelity_error / torch.norm(B, p='fro').item()

            # Combined score (normalized)
            score = relative_fidelity + lam * symmetry_error / torch.norm(T, p='fro').item()

        print(f"  Final fidelity error: {fidelity_error:.6f} (relative: {relative_fidelity:.6f})")
        print(f"  Final symmetry error: {symmetry_error:.6f}")
        print(f"  Combined score: {score:.6f}")
        print(f"  Time: {time.time() - start:.2f}s")

        # Store results
        results[lam] = {
            'fidelity_error': fidelity_error,
            'relative_fidelity': relative_fidelity,
            'symmetry_error': symmetry_error,
            'combined_score': score,
            'loss_history': [float(x) for x in loss_history[-10:]],  # Last 10 values
            'time': time.time() - start
        }

        # Update best
        if score < best_score:
            best_score = score
            best_lambda = lam
            best_T = T.detach().clone()

    print(f"\n=== Best Lambda: {best_lambda} ===")
    print(f"Best combined score: {best_score:.6f}")

    return best_T, best_lambda, results


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """
    Compute Structural Similarity Index (SSIM) for MNIST images.

    Args:
        img1: Image 1 (28x28)
        img2: Image 2 (28x28)
        window_size: Size of Gaussian window (default 11)

    Returns:
        SSIM value (higher is better, max 1.0)
    """
    # Ensure images are 2D
    if img1.dim() == 1:
        img1 = img1.view(28, 28)
    if img2.dim() == 1:
        img2 = img2.view(28, 28)

    # Constants for stability
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # Compute means
    mu1 = img1.mean()
    mu2 = img2.mean()

    # Compute variances and covariance
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    # SSIM formula
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim = numerator / denominator
    return ssim.item()


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 255.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) for MNIST images.

    Args:
        img1: Image 1 (28x28)
        img2: Image 2 (28x28)
        max_val: Maximum pixel value (default 255)

    Returns:
        PSNR value in dB (higher is better)
    """
    mse = ((img1 - img2) ** 2).mean().item()
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return psnr


def evaluate_reconstruction(
    A: torch.Tensor,
    B: torch.Tensor,
    T: torch.Tensor,
    name: str = "T"
) -> Dict:
    """
    Evaluate reconstruction quality using SSIM and PSNR.

    Args:
        A: Feature matrix (n_samples, d_A)
        B: Original pixel matrix (n_samples, d_B)
        T: Transition matrix (d_B, d_A)
        name: Name of the transition matrix

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n=== Evaluating Reconstruction ({name}) ===")
    start = time.time()

    # Reconstruct
    B_recon = A @ T.T

    n_samples = B.shape[0]
    ssim_values = []
    psnr_values = []

    print("Computing SSIM and PSNR for each sample...")
    for i in range(n_samples):
        if i % 500 == 0:
            print(f"  Progress: {i}/{n_samples}")

        # Get original and reconstructed images
        img_orig = B[i] * 255  # Scale to [0, 255]
        img_recon = B_recon[i] * 255

        # Compute metrics
        ssim_val = compute_ssim(img_orig, img_recon)
        psnr_val = compute_psnr(img_orig, img_recon)

        ssim_values.append(ssim_val)
        psnr_values.append(psnr_val)

    # Aggregate statistics
    results = {
        'ssim_mean': float(np.mean(ssim_values)),
        'ssim_std': float(np.std(ssim_values)),
        'ssim_median': float(np.median(ssim_values)),
        'psnr_mean': float(np.mean(psnr_values)),
        'psnr_std': float(np.std(psnr_values)),
        'psnr_median': float(np.median(psnr_values)),
        'fidelity_error': torch.norm(B - B_recon, p='fro').item(),
        'relative_error': (torch.norm(B - B_recon, p='fro') / torch.norm(B, p='fro')).item()
    }

    print(f"SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
    print(f"PSNR: {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB")
    print(f"Fidelity error: {results['fidelity_error']:.6f}")
    print(f"Relative error: {results['relative_error']:.6f}")
    print(f"Evaluation time: {time.time() - start:.2f}s")

    return results


def compute_symmetry_error(T: torch.Tensor, JA: torch.Tensor, JB: torch.Tensor) -> float:
    """
    Compute symmetry error: ||T J^A - J^B T||_F

    Args:
        T: Transition matrix (d_B, d_A)
        JA: Generator for A space (d_A, d_A)
        JB: Generator for B space (d_B, d_B)

    Returns:
        Symmetry error (Frobenius norm)
    """
    error = torch.norm(T @ JA - JB @ T, p='fro').item()
    return error


def differentiable_rotate(images: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """
    Differentiable image rotation using affine transformation.

    Args:
        images: Batch of images (n_samples, 784)
        angle_deg: Rotation angle in degrees

    Returns:
        Rotated images (n_samples, 784)
    """
    n_samples = images.shape[0]

    # Reshape to (n_samples, 1, 28, 28)
    imgs = images.view(n_samples, 1, 28, 28)

    # Convert angle to radians
    angle_rad = torch.tensor(angle_deg * np.pi / 180.0, device=images.device)

    # Create rotation matrix
    cos_theta = torch.cos(angle_rad)
    sin_theta = torch.sin(angle_rad)

    # Affine matrix for rotation (2x3)
    theta = torch.zeros((n_samples, 2, 3), device=images.device)
    theta[:, 0, 0] = cos_theta
    theta[:, 0, 1] = -sin_theta
    theta[:, 1, 0] = sin_theta
    theta[:, 1, 1] = cos_theta

    # Apply affine transformation
    grid = torch.nn.functional.affine_grid(theta, imgs.size(), align_corners=False)
    rotated = torch.nn.functional.grid_sample(imgs, grid, mode='bilinear',
                                               padding_mode='zeros', align_corners=False)

    # Reshape back to (n_samples, 784)
    return rotated.view(n_samples, -1)


def robustness_test(
    A: torch.Tensor,
    B: torch.Tensor,
    T_old: torch.Tensor,
    T_new: torch.Tensor,
    angles: list = [-30, -15, 15, 30]
) -> Dict:
    """
    Test robustness of T_old and T_new under image rotations.

    Args:
        A: Feature matrix (n_samples, d_A) - features from model
        B: Pixel matrix (n_samples, d_B) - original images
        T_old: Baseline transition matrix
        T_new: Equivariant transition matrix
        angles: List of rotation angles in degrees

    Returns:
        Dictionary with robustness test results
    """
    print("\n=== Robustness Test ===")
    print(f"Testing angles: {angles}")

    # Load the trained model to extract features from rotated images
    model_path = Path('/app/sandbox/session_20260120_162040_ff659490feac/outputs/mnist/models/mnist_cnn_best.pt')

    # CNN architecture matching Step 4
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(6272, 490)  # Feature layer
            self.fc2 = nn.Linear(490, 10)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool1(x)
            x = torch.relu(self.conv3(x))
            x = self.pool2(x)
            x = torch.flatten(x, 1)
            features = torch.relu(self.fc1(x))
            x = self.dropout(features)
            x = self.fc2(x)
            return x, features

    # Load model
    model = SimpleCNN().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    results = {}

    for angle in angles:
        print(f"\n--- Angle: {angle}° ---")
        start = time.time()

        # Rotate images
        B_rot = differentiable_rotate(B, angle)

        # Extract features from rotated images
        with torch.no_grad():
            B_rot_images = B_rot.view(-1, 1, 28, 28)
            _, A_rot = model(B_rot_images)

        # Predict using T_old and T_new
        B_pred_old = A_rot @ T_old.T
        B_pred_new = A_rot @ T_new.T

        # Evaluate predictions
        n_samples = min(500, B_rot.shape[0])  # Subsample for efficiency
        ssim_old_list = []
        ssim_new_list = []
        psnr_old_list = []
        psnr_new_list = []

        print(f"  Computing metrics for {n_samples} samples...")
        for i in range(n_samples):
            if i % 100 == 0:
                print(f"    Progress: {i}/{n_samples}")

            img_rot = B_rot[i] * 255
            pred_old = B_pred_old[i] * 255
            pred_new = B_pred_new[i] * 255

            ssim_old_list.append(compute_ssim(img_rot, pred_old))
            ssim_new_list.append(compute_ssim(img_rot, pred_new))
            psnr_old_list.append(compute_psnr(img_rot, pred_old))
            psnr_new_list.append(compute_psnr(img_rot, pred_new))

        results[angle] = {
            'T_old': {
                'ssim_mean': float(np.mean(ssim_old_list)),
                'ssim_std': float(np.std(ssim_old_list)),
                'psnr_mean': float(np.mean(psnr_old_list)),
                'psnr_std': float(np.std(psnr_old_list)),
                'fidelity_error': torch.norm(B_rot - B_pred_old, p='fro').item()
            },
            'T_new': {
                'ssim_mean': float(np.mean(ssim_new_list)),
                'ssim_std': float(np.std(ssim_new_list)),
                'psnr_mean': float(np.mean(psnr_new_list)),
                'psnr_std': float(np.std(psnr_new_list)),
                'fidelity_error': torch.norm(B_rot - B_pred_new, p='fro').item()
            },
            'improvement': {
                'ssim_diff': float(np.mean(ssim_new_list) - np.mean(ssim_old_list)),
                'psnr_diff': float(np.mean(psnr_new_list) - np.mean(psnr_old_list))
            },
            'time': time.time() - start
        }

        print(f"  T_old - SSIM: {results[angle]['T_old']['ssim_mean']:.4f}, "
              f"PSNR: {results[angle]['T_old']['psnr_mean']:.2f} dB")
        print(f"  T_new - SSIM: {results[angle]['T_new']['ssim_mean']:.4f}, "
              f"PSNR: {results[angle]['T_new']['psnr_mean']:.2f} dB")
        print(f"  Improvement - SSIM: {results[angle]['improvement']['ssim_diff']:+.4f}, "
              f"PSNR: {results[angle]['improvement']['psnr_diff']:+.2f} dB")
        print(f"  Time: {results[angle]['time']:.2f}s")

    return results


def main():
    """Main execution function."""
    print("="*80)
    print("Step 6: Large-Scale Equivariant Optimization (MNIST)")
    print("="*80)

    overall_start = time.time()

    # 1. Load matrices
    A, B, JA, JB = load_matrices()

    # 2. Compute baseline T_old
    T_old = compute_baseline_T_old(A, B)

    # 3. Compute equivariant T_new
    T_new, best_lambda, lambda_sweep_results = compute_equivariant_T_new(
        A, B, JA, JB,
        lambda_values=[0.1, 1.0, 10.0]
    )

    # 4. Evaluate reconstruction quality
    eval_old = evaluate_reconstruction(A, B, T_old, name="T_old")
    eval_new = evaluate_reconstruction(A, B, T_new, name="T_new")

    # 5. Compute symmetry errors
    print("\n=== Symmetry Errors ===")
    symmetry_old = compute_symmetry_error(T_old, JA, JB)
    symmetry_new = compute_symmetry_error(T_new, JA, JB)
    print(f"T_old symmetry error: {symmetry_old:.6f}")
    print(f"T_new symmetry error: {symmetry_new:.6f}")
    print(f"Improvement: {symmetry_old - symmetry_new:.6f}")

    # 6. Robustness test
    robustness_results = robustness_test(A, B, T_old, T_new, angles=[-30, -15, 15, 30])

    # 7. Save results
    print("\n=== Saving Outputs ===")

    # Save transition matrices
    np.save(OUTPUT_DIR / 'T_old.npy', T_old.cpu().numpy())
    np.save(OUTPUT_DIR / 'T_new.npy', T_new.cpu().numpy())
    print(f"Saved: {OUTPUT_DIR / 'T_old.npy'}")
    print(f"Saved: {OUTPUT_DIR / 'T_new.npy'}")

    # Compile all results
    all_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(device),
        'matrices': {
            'A_shape': list(A.shape),
            'B_shape': list(B.shape),
            'JA_shape': list(JA.shape),
            'JB_shape': list(JB.shape),
            'T_old_shape': list(T_old.shape),
            'T_new_shape': list(T_new.shape)
        },
        'optimization': {
            'best_lambda': float(best_lambda),
            'lambda_sweep': lambda_sweep_results
        },
        'evaluation': {
            'T_old': eval_old,
            'T_new': eval_new,
            'improvement': {
                'ssim_diff': eval_new['ssim_mean'] - eval_old['ssim_mean'],
                'psnr_diff': eval_new['psnr_mean'] - eval_old['psnr_mean'],
                'relative_error_diff': eval_old['relative_error'] - eval_new['relative_error']
            }
        },
        'symmetry': {
            'T_old': float(symmetry_old),
            'T_new': float(symmetry_new),
            'improvement': float(symmetry_old - symmetry_new)
        },
        'robustness': robustness_results,
        'total_time': time.time() - overall_start
    }

    # Save results JSON
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {RESULTS_FILE}")

    # Print summary
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    print(f"\nBest lambda: {best_lambda}")
    print(f"\nReconstruction Quality:")
    print(f"  T_old - SSIM: {eval_old['ssim_mean']:.4f}, PSNR: {eval_old['psnr_mean']:.2f} dB")
    print(f"  T_new - SSIM: {eval_new['ssim_mean']:.4f}, PSNR: {eval_new['psnr_mean']:.2f} dB")
    print(f"  Improvement: SSIM {all_results['evaluation']['improvement']['ssim_diff']:+.4f}, "
          f"PSNR {all_results['evaluation']['improvement']['psnr_diff']:+.2f} dB")
    print(f"\nSymmetry Error:")
    print(f"  T_old: {symmetry_old:.6f}")
    print(f"  T_new: {symmetry_new:.6f}")
    print(f"  Reduction: {all_results['symmetry']['improvement']:.6f} "
          f"({100*all_results['symmetry']['improvement']/symmetry_old:.1f}%)")
    print(f"\nRobustness (average over all angles):")
    avg_ssim_old = np.mean([robustness_results[a]['T_old']['ssim_mean'] for a in robustness_results])
    avg_ssim_new = np.mean([robustness_results[a]['T_new']['ssim_mean'] for a in robustness_results])
    avg_psnr_old = np.mean([robustness_results[a]['T_old']['psnr_mean'] for a in robustness_results])
    avg_psnr_new = np.mean([robustness_results[a]['T_new']['psnr_mean'] for a in robustness_results])
    print(f"  T_old - SSIM: {avg_ssim_old:.4f}, PSNR: {avg_psnr_old:.2f} dB")
    print(f"  T_new - SSIM: {avg_ssim_new:.4f}, PSNR: {avg_psnr_new:.2f} dB")
    print(f"  Improvement: SSIM {avg_ssim_new - avg_ssim_old:+.4f}, "
          f"PSNR {avg_psnr_new - avg_psnr_old:+.2f} dB")
    print(f"\nTotal execution time: {all_results['total_time']:.2f}s ({all_results['total_time']/60:.1f} min)")
    print("\n" + "="*80)
    print("Step 6 completed successfully!")
    print("="*80)


if __name__ == '__main__':
    main()

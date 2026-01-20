#!/usr/bin/env python3
"""
Script 02: Synthetic Domain Experiments - Baseline Validation & Generator Estimation
======================================================================================

This script implements:
1. Baseline validation: Compute reconstruction fidelity using T_old
2. Algorithm 2: Lie algebra generator estimation via MDS + rotation

Author: K-Dense Coding Agent
Date: 2026-01-20
"""

import json
import numpy as np
from pathlib import Path
from sklearn.manifold import MDS
from sklearn.linear_model import LinearRegression
import yaml

# Set random seed for reproducibility
np.random.seed(42)

def load_config():
    """Load configuration from config.yaml."""
    config_path = Path("/app/sandbox/session_20260120_162040_ff659490feac/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_matrices(data_path):
    """
    Load matrices A, B, and T_old from JSON files.

    Parameters:
    -----------
    data_path : Path
        Path to directory containing A.json, B.json, T_old.json

    Returns:
    --------
    A, B, T_old : numpy arrays
    """
    with open(data_path / "A.json", 'r') as f:
        data_A = json.load(f)
        A = np.array(data_A['data'])

    with open(data_path / "B.json", 'r') as f:
        data_B = json.load(f)
        B = np.array(data_B['data'])

    with open(data_path / "T_old.json", 'r') as f:
        data_T = json.load(f)
        T_old = np.array(data_T['data'])

    print(f"  Loaded A: {A.shape}")
    print(f"  Loaded B: {B.shape}")
    print(f"  Loaded T_old: {T_old.shape}")

    return A, B, T_old

def baseline_validation(A, B, T_old):
    """
    Perform baseline validation.

    Compute reconstruction B* = A @ T_old and calculate MSE.

    Parameters:
    -----------
    A : np.ndarray (m x n)
        Matrix A (15 x 5)
    B : np.ndarray (m x l)
        Matrix B (15 x 4)
    T_old : np.ndarray (n x l)
        Transformation matrix T_old (5 x 4)

    Returns:
    --------
    mse : float
        Mean squared error of reconstruction
    B_reconstructed : np.ndarray
        Reconstructed B* (15 x 4)
    """
    # Compute B* = A @ T_old (matrix multiplication: (15,5) @ (5,4) = (15,4))
    B_reconstructed = A @ T_old

    # Calculate MSE: (1/(m*l)) * ||B - B*||_F^2
    m, l = B.shape
    frobenius_norm_squared = np.sum((B - B_reconstructed) ** 2)
    mse = frobenius_norm_squared / (m * l)

    print(f"  Baseline Fidelity MSE: {mse:.8e}")

    return mse, B_reconstructed

def estimate_generator(X, epsilon=0.01):
    """
    Estimate Lie algebra generator using Algorithm 2.

    Steps:
    1. Reduce X to 2D using MDS
    2. Train Linear Regression decoder (2D -> original dimension)
    3. Rotate 2D representation by epsilon
    4. Decode rotated representation
    5. Compute numerical derivative
    6. Solve for generator J via least squares

    Parameters:
    -----------
    X : np.ndarray (m x d)
        Input matrix (A or B)
    epsilon : float
        Rotation angle in radians (default: 0.01)

    Returns:
    --------
    J : np.ndarray (d x d)
        Estimated Lie algebra generator
    """
    m, d = X.shape
    print(f"  Estimating generator for {X.shape} matrix")

    # Step 1: MDS reduction to 2D
    print(f"    Step 1/5: Performing MDS reduction to 2D...")
    mds = MDS(n_components=2, random_state=42, dissimilarity='euclidean')
    X_2d = mds.fit_transform(X)
    print(f"    MDS complete. Stress: {mds.stress_:.4f}")

    # Step 2: Train Linear Regression decoder (2D -> d)
    print(f"    Step 2/5: Training Linear Regression decoder (2D -> {d}D)...")
    decoder = LinearRegression()
    decoder.fit(X_2d, X)

    # Compute reconstruction error
    X_reconstructed = decoder.predict(X_2d)
    reconstruction_mse = np.mean((X - X_reconstructed) ** 2)
    print(f"    Decoder MSE: {reconstruction_mse:.8e}")

    # Step 3: Rotate X_2d by epsilon
    print(f"    Step 3/5: Rotating 2D representation by ε={epsilon} rad...")
    rotation_matrix = np.array([
        [np.cos(epsilon), -np.sin(epsilon)],
        [np.sin(epsilon),  np.cos(epsilon)]
    ])
    X_2d_rot = X_2d @ rotation_matrix.T

    # Step 4: Decode rotated representation
    print(f"    Step 4/5: Decoding rotated representation...")
    X_rot = decoder.predict(X_2d_rot)

    # Step 5: Compute numerical derivative
    print(f"    Step 5/5: Computing numerical derivative and solving for generator...")
    Delta_X = (X_rot - X) / epsilon

    # Solve for J: X @ J^T ≈ Delta_X
    # This is equivalent to: J^T = (X^T X)^{-1} X^T Delta_X
    # Or in standard form: J = Delta_X^T X (X^T X)^{-1}
    # Using least squares: J^T ≈ X \ Delta_X (solve X @ J^T = Delta_X)

    # Method: lstsq for each row of J
    J = np.zeros((d, d))
    for i in range(d):
        # Solve: X @ J[i, :] = Delta_X[:, i]
        J[i, :], _, _, _ = np.linalg.lstsq(X, Delta_X[:, i], rcond=None)

    print(f"    Generator J estimated: {J.shape}")
    print(f"    Generator norm: {np.linalg.norm(J):.6f}")

    return J

def process_dataset(data_type, config):
    """
    Process a single dataset (primary or sensitivity).

    Parameters:
    -----------
    data_type : str
        Either 'primary' or 'sensitivity'
    config : dict
        Configuration dictionary
    """
    print(f"\n{'='*80}")
    print(f"Processing: {data_type.upper()} dataset")
    print(f"{'='*80}")

    # Load matrices
    data_path = Path("/app/sandbox/session_20260120_162040_ff659490feac") / config['paths']['inputs'][f'synthetic_{data_type}']
    A, B, T_old = load_matrices(data_path)

    # Baseline validation
    print(f"\n--- Baseline Validation ---")
    mse, B_reconstructed = baseline_validation(A, B, T_old)

    # Generator estimation for A
    print(f"\n--- Generator Estimation for A ---")
    epsilon = 0.01  # Rotation angle for Algorithm 2
    J_A = estimate_generator(A, epsilon=epsilon)

    # Generator estimation for B
    print(f"\n--- Generator Estimation for B ---")
    J_B = estimate_generator(B, epsilon=epsilon)

    # Save outputs
    output_dir = Path("/app/sandbox/session_20260120_162040_ff659490feac/outputs/synthetic") / data_type / "matrices"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save J_A
    J_A_path = output_dir / "JA.json"
    with open(J_A_path, 'w') as f:
        json.dump({
            'data': J_A.tolist(),
            'shape': J_A.shape,
            'source': f'{data_type} dataset',
            'epsilon': epsilon,
            'description': 'Lie algebra generator for matrix A'
        }, f, indent=2)
    print(f"\n✓ Saved J_A to {J_A_path}")

    # Save J_B
    J_B_path = output_dir / "JB.json"
    with open(J_B_path, 'w') as f:
        json.dump({
            'data': J_B.tolist(),
            'shape': J_B.shape,
            'source': f'{data_type} dataset',
            'epsilon': epsilon,
            'description': 'Lie algebra generator for matrix B'
        }, f, indent=2)
    print(f"✓ Saved J_B to {J_B_path}")

    # Log metrics
    log_file = Path("/app/sandbox/session_20260120_162040_ff659490feac/outputs/logs/synthetic_step2_metrics.txt")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Dataset: {data_type}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Baseline Fidelity MSE: {mse:.8e}\n")
        f.write(f"Matrix A shape: {A.shape}\n")
        f.write(f"Matrix B shape: {B.shape}\n")
        f.write(f"Generator J_A shape: {J_A.shape}\n")
        f.write(f"Generator J_B shape: {J_B.shape}\n")
        f.write(f"Generator J_A norm: {np.linalg.norm(J_A):.6f}\n")
        f.write(f"Generator J_B norm: {np.linalg.norm(J_B):.6f}\n")
        f.write(f"Epsilon (rotation angle): {epsilon} rad\n")

    print(f"✓ Appended metrics to {log_file}")

    return {
        'mse': mse,
        'J_A': J_A,
        'J_B': J_B,
        'J_A_path': str(J_A_path),
        'J_B_path': str(J_B_path)
    }

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("STEP 2: SYNTHETIC DOMAIN EXPERIMENTS")
    print("Baseline Validation & Generator Estimation (Algorithm 2)")
    print("="*80)

    # Load configuration
    config = load_config()
    print(f"\n✓ Configuration loaded from config.yaml")

    # Process both datasets
    results = {}
    for data_type in ['primary', 'sensitivity']:
        results[data_type] = process_dataset(data_type, config)

    # Final summary
    print("\n" + "="*80)
    print("EXECUTION COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    for data_type in ['primary', 'sensitivity']:
        print(f"\n{data_type.upper()}:")
        print(f"  - {results[data_type]['J_A_path']}")
        print(f"  - {results[data_type]['J_B_path']}")

    print(f"\n  - /app/sandbox/session_20260120_162040_ff659490feac/outputs/logs/synthetic_step2_metrics.txt")

    print("\nBaseline MSE comparison:")
    print(f"  Primary:    {results['primary']['mse']:.8e}")
    print(f"  Sensitivity: {results['sensitivity']['mse']:.8e}")

    print("\n✓ All outputs generated successfully!")

if __name__ == "__main__":
    main()

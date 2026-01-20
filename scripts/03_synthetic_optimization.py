#!/usr/bin/env python3
"""
Step 3: Synthetic Domain Experiments - Equivariant Optimization & Robustness Testing

This script implements:
1. Algorithm 1 (Equivariant Optimization) using Kronecker products
2. Lambda regularization sweep
3. Robustness test (Scenario 3) with rotation generation

Author: K-Dense Coding Agent
"""

import json
import numpy as np
from pathlib import Path
from sklearn.manifold import MDS
from sklearn.linear_model import LinearRegression
from scipy.linalg import lstsq
import sys

# Set random seed for reproducibility
np.random.seed(42)

# Base paths
BASE_DIR = Path("/app/sandbox/session_20260120_162040_ff659490feac")
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Ensure directories exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def load_matrices(dataset_type):
    """Load all matrices for a dataset type (primary or sensitivity)."""
    print(f"\n{'='*60}")
    print(f"Loading matrices for {dataset_type} dataset...")
    print(f"{'='*60}")

    # Input matrices are in inputs/synthetic, generators are in outputs/synthetic
    inputs_path = BASE_DIR / "inputs" / "synthetic" / dataset_type
    outputs_path = OUTPUTS_DIR / "synthetic" / dataset_type / "matrices"

    # Load input matrices (structured format with 'data' field)
    with open(inputs_path / "A.json", "r") as f:
        A_json = json.load(f)
        A = np.array(A_json['data'] if isinstance(A_json, dict) and 'data' in A_json else A_json)
    with open(inputs_path / "B.json", "r") as f:
        B_json = json.load(f)
        B = np.array(B_json['data'] if isinstance(B_json, dict) and 'data' in B_json else B_json)
    with open(inputs_path / "T_old.json", "r") as f:
        T_old_json = json.load(f)
        T_old = np.array(T_old_json['data'] if isinstance(T_old_json, dict) and 'data' in T_old_json else T_old_json)

    # Load generators (from previous step's outputs - also structured format)
    with open(outputs_path / "JA.json", "r") as f:
        JA_json = json.load(f)
        JA = np.array(JA_json['data'] if isinstance(JA_json, dict) and 'data' in JA_json else JA_json)
    with open(outputs_path / "JB.json", "r") as f:
        JB_json = json.load(f)
        JB = np.array(JB_json['data'] if isinstance(JB_json, dict) and 'data' in JB_json else JB_json)

    print(f"Loaded matrices:")
    print(f"  A: {A.shape}")
    print(f"  B: {B.shape}")
    print(f"  T_old: {T_old.shape}")
    print(f"  JA: {JA.shape}")
    print(f"  JB: {JB.shape}")

    return A, B, T_old, JA, JB


def algorithm_1_equivariant_optimization(A, B, JA, JB, lambda_val, tau=1e-10):
    """
    Algorithm 1: Equivariant Optimization using Kronecker products.

    Constructs the linear system M * vec(T) = Y where:
    - Fidelity term: (A ⊗ I_l) targeting vec(B^T)
    - Symmetry term: λ((J^A)^T ⊗ I_l - I_k ⊗ J^B) targeting 0

    Solves using SVD-based pseudoinverse with singular value truncation.

    Args:
        A: Source matrix (m x k)
        B: Target matrix (m x l)
        JA: Generator for space A (k x k)
        JB: Generator for space B (l x l)
        lambda_val: Regularization parameter
        tau: Singular value threshold for pseudoinverse

    Returns:
        T: Transition matrix (l x k)
        info: Dictionary with optimization metrics
    """
    m, k = A.shape
    _, l = B.shape

    print(f"\n{'='*60}")
    print(f"Algorithm 1: Equivariant Optimization (λ={lambda_val})")
    print(f"{'='*60}")
    print(f"Input shapes: A={A.shape}, B={B.shape}, JA={JA.shape}, JB={JB.shape}")

    # Create identity matrices
    I_k = np.eye(k)
    I_l = np.eye(l)

    # Fidelity term: A ⊗ I_l
    # Using Kronecker product to construct the fidelity constraint
    print(f"Constructing fidelity term (A ⊗ I_l)...")
    M_fidelity = np.kron(A, I_l)  # Shape: (m*l, k*l)
    Y_fidelity = B.T.flatten()     # vec(B^T), shape: (l*m,) = (k*l,) wait, needs to be (m*l,)
    # Actually vec(B^T) should be (l*m), but we want to target B, so vec(B^T)
    # Let me reconsider: If T is (l x k), then A @ T.T gives (m x k) @ (k x l) = (m x l)
    # So B* = A @ W where W = T.T (k x l)
    # We want to solve for T such that B ≈ A @ T.T
    # In vectorized form: vec(B) ≈ vec(A @ T.T) = (T ⊗ I_m) @ vec(A^T) -- NO
    # Let me use the standard identity: vec(A @ X @ B) = (B^T ⊗ A) @ vec(X)
    # Here: B = A @ W where W = T.T (k x l), so vec(B) = (I_l^T ⊗ A) @ vec(W) = (I_l ⊗ A) @ vec(T.T)
    # vec(T.T) = vec(T) under row-major (but we need column-major)
    # In column-major: vec(T.T) relates to vec(T)
    # Let's use: B = A @ T.T, so B^T = T @ A^T
    # vec(B^T) = (A ⊗ I_l) @ vec(T)
    # So fidelity: (A ⊗ I_l) @ vec(T) = vec(B^T)

    Y_fidelity = B.T.flatten('F')  # vec(B^T) in column-major order, shape: (l*m,)
    # But M_fidelity is (m*l, k*l), and vec(T) is (l*k,)
    # Kronecker product (A ⊗ I_l) should be (m*l, k*l) ✓
    # So this should be: (m*l, k*l) @ (k*l, 1) = (m*l, 1) ✓

    print(f"  M_fidelity shape: {M_fidelity.shape}")
    print(f"  Y_fidelity shape: {Y_fidelity.shape}")

    # Symmetry term: λ((J^A)^T ⊗ I_l - I_k ⊗ J^B)
    # This enforces T @ J^A = J^B @ T
    # vec(T @ J^A) = (J^A^T ⊗ I_l) @ vec(T)
    # vec(J^B @ T) = (I_k ⊗ J^B) @ vec(T)
    # So: ((J^A)^T ⊗ I_l - I_k ⊗ J^B) @ vec(T) = 0
    print(f"Constructing symmetry term λ((J^A)^T ⊗ I_l - I_k ⊗ J^B)...")
    term1 = np.kron(JA.T, I_l)     # Shape: (k*l, k*l)
    term2 = np.kron(I_k, JB)       # Shape: (k*l, k*l)
    M_symmetry = lambda_val * (term1 - term2)  # Shape: (k*l, k*l)
    Y_symmetry = np.zeros(k * l)   # Target: 0 vector

    print(f"  M_symmetry shape: {M_symmetry.shape}")
    print(f"  Y_symmetry shape: {Y_symmetry.shape}")

    # Stack constraints
    print(f"Stacking constraints...")
    M = np.vstack([M_fidelity, M_symmetry])  # Shape: (m*l + k*l, k*l)
    Y = np.concatenate([Y_fidelity, Y_symmetry])  # Shape: (m*l + k*l,)

    print(f"  Full system M shape: {M.shape}")
    print(f"  Full system Y shape: {Y.shape}")

    # Solve using SVD-based pseudoinverse
    print(f"Computing SVD...")
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    print(f"  Singular values range: [{s.min():.2e}, {s.max():.2e}]")
    print(f"  Number of singular values > τ={tau}: {np.sum(s > tau)}/{len(s)}")

    # Truncate small singular values
    s_inv = np.where(s > tau, 1.0 / s, 0.0)

    # Compute pseudoinverse solution: vec(T) = V @ diag(s_inv) @ U^T @ Y
    print(f"Computing pseudoinverse solution...")
    vec_T = Vt.T @ (s_inv[:, np.newaxis] * (U.T @ Y[:, np.newaxis]))
    vec_T = vec_T.flatten()

    # Reshape to get T (l x k)
    T = vec_T.reshape((l, k), order='F')  # Column-major order

    print(f"Solution T shape: {T.shape}")

    # Compute metrics
    print(f"Computing metrics...")
    # Fidelity error: ||B - A @ T.T||_F^2 / (m*l)
    B_pred = A @ T.T
    MSE_fid = np.sum((B - B_pred)**2) / (m * l)

    # Symmetry error: ||T @ J^A - J^B @ T||_F^2
    Sym_err = np.sum((T @ JA - JB @ T)**2)

    print(f"  MSE_fidelity: {MSE_fid:.6e}")
    print(f"  Sym_error: {Sym_err:.6e}")

    info = {
        'MSE_fid': float(MSE_fid),
        'Sym_err': float(Sym_err),
        'lambda': float(lambda_val),
        'n_singular_values': int(np.sum(s > tau)),
        'condition_number': float(s.max() / max(s[s > tau].min(), tau))
    }

    return T, info


def lambda_sweep(A, B, JA, JB, lambda_values, dataset_type):
    """
    Perform lambda sweep: compute T_new for multiple lambda values.

    Args:
        A, B: Input matrices
        JA, JB: Generators
        lambda_values: List of lambda values to test
        dataset_type: 'primary' or 'sensitivity'

    Returns:
        results: Dictionary with sweep results
        T_final: T_new for lambda=0.5
    """
    print(f"\n{'='*60}")
    print(f"Lambda Sweep for {dataset_type} dataset")
    print(f"{'='*60}")
    print(f"Testing λ values: {lambda_values}")

    results = []
    T_final = None

    for i, lambda_val in enumerate(lambda_values):
        print(f"\n[{i+1}/{len(lambda_values)}] Processing λ={lambda_val}...")

        T, info = algorithm_1_equivariant_optimization(A, B, JA, JB, lambda_val)

        result = {
            'dataset_type': dataset_type,
            'lambda': float(lambda_val),
            'MSE_fid': info['MSE_fid'],
            'Sym_err': info['Sym_err'],
            'n_singular_values': info['n_singular_values'],
            'condition_number': info['condition_number']
        }
        results.append(result)

        # Save T for lambda=0.5 as final solution
        if lambda_val == 0.5:
            T_final = T
            print(f"  → Saving this as T_final (λ=0.5)")

    return results, T_final


def rotation_matrix_2d(angle_deg):
    """Create 2D rotation matrix for given angle in degrees."""
    theta = np.radians(angle_deg)
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])


def robustness_test_scenario3(A, B, T_old, T_new, dataset_type,
                               angle_range=(-30, 30), angle_step=5):
    """
    Robustness Test (Scenario 3): Test equivariance under rotations.

    Pipeline: MDS → Decoder → Rotate → Encode → Compare

    For each angle α:
    1. A_rot(α), B_target(α) = MDS → Decoder → Rotate(α) → Encode
    2. B*_old = A_rot @ W_old (where W_old = T_old)
    3. B*_new = A_rot @ W_new (where W_new = T_new.T)
    4. Compute MSE against B_target for both methods

    Args:
        A, B: Original matrices
        T_old: Old transition matrix (k x l) - wait, T_old is (5x4) from previous, so (k x l)
        T_new: New equivariant transition matrix (l x k)
        dataset_type: 'primary' or 'sensitivity'
        angle_range: (min, max) angles in degrees
        angle_step: Step size for angles

    Returns:
        results: Dictionary with robustness test data
    """
    print(f"\n{'='*60}")
    print(f"Robustness Test (Scenario 3) for {dataset_type} dataset")
    print(f"{'='*60}")

    m, k = A.shape
    _, l = B.shape

    # Note: T_old is (k x l), so W_old = T_old for multiplication A @ W_old
    # T_new is (l x k), so W_new = T_new.T for multiplication A @ W_new
    W_old = T_old  # Shape: (k x l)
    W_new = T_new.T  # Shape: (k x l)

    print(f"Matrix shapes:")
    print(f"  A: {A.shape}, B: {B.shape}")
    print(f"  T_old (W_old): {T_old.shape}")
    print(f"  T_new: {T_new.shape}, W_new: {W_new.shape}")

    # Step 1: Fit MDS to reduce A to 2D
    print(f"\nStep 1: Fitting MDS to A...")
    mds = MDS(n_components=2, random_state=42, dissimilarity='euclidean', n_init=10, max_iter=300)
    X_A = mds.fit_transform(A)
    stress_A = mds.stress_
    print(f"  MDS stress for A: {stress_A:.6f}")
    print(f"  X_A shape: {X_A.shape}")

    # Step 2: Fit decoder from 2D latent space back to k-dimensional space
    print(f"\nStep 2: Training decoder (2D → {k}D)...")
    decoder = LinearRegression()
    decoder.fit(X_A, A)
    A_reconstructed = decoder.predict(X_A)
    decoder_mse = np.mean((A - A_reconstructed)**2)
    print(f"  Decoder MSE: {decoder_mse:.6e}")

    # Step 3: Fit MDS to B for target generation
    print(f"\nStep 3: Fitting MDS to B...")
    mds_B = MDS(n_components=2, random_state=42, dissimilarity='euclidean', n_init=10, max_iter=300)
    X_B = mds_B.fit_transform(B)
    stress_B = mds_B.stress_
    print(f"  MDS stress for B: {stress_B:.6f}")
    print(f"  X_B shape: {X_B.shape}")

    # Step 4: Fit decoder for B
    print(f"\nStep 4: Training decoder (2D → {l}D)...")
    decoder_B = LinearRegression()
    decoder_B.fit(X_B, B)
    B_reconstructed = decoder_B.predict(X_B)
    decoder_B_mse = np.mean((B - B_reconstructed)**2)
    print(f"  Decoder MSE: {decoder_B_mse:.6e}")

    # Step 5: Generate rotations and test
    angles = list(range(angle_range[0], angle_range[1] + 1, angle_step))
    print(f"\nStep 5: Testing {len(angles)} rotation angles: {angles}")

    results = {
        'dataset_type': dataset_type,
        'angles': angles,
        'mse_old': [],
        'mse_new': [],
        'A_rot_coords': [],  # For visualization
        'B_target_coords': [],  # For visualization
        'B_pred_old_coords': [],
        'B_pred_new_coords': [],
        'mds_stress_A': float(stress_A),
        'mds_stress_B': float(stress_B),
        'decoder_mse_A': float(decoder_mse),
        'decoder_mse_B': float(decoder_B_mse)
    }

    for i, angle in enumerate(angles):
        if i % 3 == 0:
            print(f"  Processing angle {angle}° ({i+1}/{len(angles)})...")

        # Rotate latent representations
        R = rotation_matrix_2d(angle)
        X_A_rot = X_A @ R.T  # Rotate A's latent representation
        X_B_rot = X_B @ R.T  # Rotate B's latent representation (for target)

        # Decode rotated representations
        A_rot = decoder.predict(X_A_rot)  # Shape: (m, k)
        B_target = decoder_B.predict(X_B_rot)  # Shape: (m, l)

        # Predictions
        B_pred_old = A_rot @ W_old  # Shape: (m, k) @ (k, l) = (m, l)
        B_pred_new = A_rot @ W_new  # Shape: (m, k) @ (k, l) = (m, l)

        # Compute MSE
        mse_old = np.mean((B_target - B_pred_old)**2)
        mse_new = np.mean((B_target - B_pred_new)**2)

        results['mse_old'].append(float(mse_old))
        results['mse_new'].append(float(mse_new))

        # Store coordinates for visualization (first 5 points for brevity)
        results['A_rot_coords'].append(A_rot[:5].tolist())
        results['B_target_coords'].append(B_target[:5].tolist())
        results['B_pred_old_coords'].append(B_pred_old[:5].tolist())
        results['B_pred_new_coords'].append(B_pred_new[:5].tolist())

    print(f"\nRobustness Test Summary:")
    print(f"  Angles tested: {len(angles)}")
    print(f"  Mean MSE (old): {np.mean(results['mse_old']):.6e}")
    print(f"  Mean MSE (new): {np.mean(results['mse_new']):.6e}")
    print(f"  Improvement ratio: {np.mean(results['mse_old']) / np.mean(results['mse_new']):.2f}x")

    return results


def main():
    """Main execution function."""
    print("="*60)
    print("Step 3: Synthetic Domain Experiments")
    print("Equivariant Optimization & Robustness Testing")
    print("="*60)

    # Lambda values for sweep
    lambda_values = [0, 0.1, 0.25, 0.5, 1.0, 2.0]

    # Process both primary and sensitivity datasets
    dataset_types = ['primary', 'sensitivity']

    all_sweep_results = []

    for dataset_type in dataset_types:
        print(f"\n{'#'*60}")
        print(f"# Processing {dataset_type.upper()} dataset")
        print(f"{'#'*60}")

        # Load matrices
        A, B, T_old, JA, JB = load_matrices(dataset_type)

        # Lambda sweep
        sweep_results, T_final = lambda_sweep(A, B, JA, JB, lambda_values, dataset_type)
        all_sweep_results.extend(sweep_results)

        # Save T_new (T_final from lambda=0.5)
        output_path = OUTPUTS_DIR / "synthetic" / dataset_type / "matrices" / "T_new.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(T_final.tolist(), f, indent=2)
        print(f"\n✓ Saved T_new to: {output_path}")

        # Robustness test (Scenario 3)
        robustness_results = robustness_test_scenario3(
            A, B, T_old, T_final, dataset_type,
            angle_range=(-30, 30), angle_step=5
        )

        # Save robustness results
        rob_output_path = OUTPUTS_DIR / "synthetic" / dataset_type / "robustness_results.json"
        with open(rob_output_path, 'w') as f:
            json.dump(robustness_results, f, indent=2)
        print(f"✓ Saved robustness results to: {rob_output_path}")

    # Save combined lambda sweep results
    sweep_output_path = LOGS_DIR / "synthetic_lambda_sweep.json"
    with open(sweep_output_path, 'w') as f:
        json.dump(all_sweep_results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"✓ Saved lambda sweep results to: {sweep_output_path}")
    print(f"{'='*60}")

    print("\n" + "="*60)
    print("SUCCESS: All tasks completed!")
    print("="*60)
    print("\nGenerated outputs:")
    print("  1. T_new matrices (λ=0.5):")
    print("     - outputs/synthetic/primary/matrices/T_new.json")
    print("     - outputs/synthetic/sensitivity/matrices/T_new.json")
    print("  2. Lambda sweep metrics:")
    print("     - outputs/logs/synthetic_lambda_sweep.json")
    print("  3. Robustness test results:")
    print("     - outputs/synthetic/primary/robustness_results.json")
    print("     - outputs/synthetic/sensitivity/robustness_results.json")
    print("="*60)


if __name__ == "__main__":
    main()

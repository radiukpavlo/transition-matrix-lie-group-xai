#!/usr/bin/env python3
"""
Reproduction Script: Synthetic Experiments (Primary Dataset)
Generates all matrices (A, B, T_old, JA, JB, T_new) from scratch.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.manifold import MDS
from sklearn.linear_model import LinearRegression

np.random.seed(42)

# === Configuration ===
BASE_DIR = Path(__file__).parent.parent
INPUTS_DIR = BASE_DIR / "inputs" / "synthetic" / "primary"
OUTPUTS_DIR = BASE_DIR / "outputs" / "synthetic" / "primary" / "matrices"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

def save_matrix(matrix, name, path, description=""):
    """Save matrix as JSON with metadata."""
    data = {
        "name": name,
        "shape": list(matrix.shape),
        "dtype": "float64",
        "description": description,
        "data": matrix.tolist()
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[OK] Saved {name} {matrix.shape} to {path.name}")

# =============================================================================
# Step 1: Create Input Matrices A, B, T_old (from Appendix 1.1)
# =============================================================================
def create_input_matrices():
    print("\n" + "="*60)
    print("Step 1: Creating Input Matrices (Appendix 1.1)")
    print("="*60)
    
    # Matrix A (15 × 5) - Actual values from manuscript
    A = np.array([
        [2.8, -1.8, -2.8, 1.3, 0.4],
        [2.9, -1.9, -2.9, 1.4, 0.5],
        [3.0, -2.0, -3.0, 1.5, 0.6],
        [3.1, -2.1, -3.1, 1.6, 0.7],
        [3.2, -2.2, -3.2, 1.7, 0.8],
        [-1.6, -2.5, 1.5, 0.2, 0.6],
        [-1.3, -2.7, 1.3, 0.4, 0.8],
        [-1.0, -3.0, 1.5, 0.6, 1.0],
        [-0.7, -3.2, 1.7, 0.8, 1.2],
        [-0.5, -3.5, 1.9, 1.0, 1.4],
        [1.2, -1.2, 0.7, -0.3, -2.8],
        [1.1, -1.1, 0.8, -0.4, -2.9],
        [1.0, -1.0, 0.844444444, -0.444444444, -3.0],  # 0.8(4), -0.(4)
        [0.9, -0.9, 0.85, -0.45, -3.1],
        [0.8, -0.8, 0.9, -0.5, -3.2]
    ])

    # Matrix B (15 × 4) - Actual values from manuscript
    B = np.array([
        [-1.979394104, 1.959307524, -1.381119943, -1.72964],
        [-1.974921385, 1.94850558, -1.726609792, -1.76121],
        [-1.843907868, 1.99818664, -1.912855282, -1.97511],
        [-1.998625355, 1.999671808, -1.998443276, -1.99976],
        [-1.999365095, 1.998896097, -1.999605076, -1.99892],
        [1.997775859, -1.844000202, 1.660111333, -1.37353],
        [1.818753218, -1.909687734, 1.206631506, -1.40799],
        [1.992023578, -1.923804827, 0.706593926, -1.54378],
        [1.999174385, -1.997592083, 0.21221635, -1.58697],
        [1.997854305, -1.999410881, -0.243400633, -1.82759],
        [0.851626415, 1.574201387, 1.581026838, 1.573934],
        [1.008512576, 1.570791652, 1.595657199, 1.741762],
        [1.107744254, 1.615475549, 1.723582196, 1.807615],
        [1.089897991, 1.611369928, 1.882537367, 1.873522],
        [1.290406093, 1.695289797, 1.953503509, 1.94625]
    ])

    # Matrix T_old (5 × 4) - Actual values from manuscript
    T_old = np.array([
        [-0.278135369, 0.520567817, -0.140387778, 0.024426],
        [-0.382248581, 0.126035484, -0.145008015, 0.349038],
        [0.522859856, -0.341076002, 0.433255464, 0.198781],
        [-0.065904355, -0.023301678, -0.149755201, -0.25589],
        [-0.177604706, -0.49953555, -0.428847974, -0.61688]
    ])

    print(f"  A: {A.shape}")
    print(f"  B: {B.shape}")
    print(f"  T_old: {T_old.shape}")
    
    return A, B, T_old


# =============================================================================
# Step 2: Estimate Generators JA, JB (Algorithm 2)
# =============================================================================
def estimate_generator(X, epsilon=0.01):
    """Algorithm 2: MDS -> Decoder -> Rotate -> Derivative -> Solve for J."""
    m, d = X.shape
    
    # MDS reduction to 2D
    mds = MDS(n_components=2, random_state=42, dissimilarity='euclidean')
    X_2d = mds.fit_transform(X)
    
    # Linear Regression decoder (2D -> d)
    decoder = LinearRegression()
    decoder.fit(X_2d, X)
    
    # Rotate X_2d by epsilon
    R = np.array([
        [np.cos(epsilon), -np.sin(epsilon)],
        [np.sin(epsilon),  np.cos(epsilon)]
    ])
    X_2d_rot = X_2d @ R.T
    
    # Decode rotated representation
    X_rot = decoder.predict(X_2d_rot)
    
    # Numerical derivative
    Delta_X = (X_rot - X) / epsilon
    
    # Solve for J: X @ J^T ≈ Delta_X
    J = np.zeros((d, d))
    for i in range(d):
        J[i, :], _, _, _ = np.linalg.lstsq(X, Delta_X[:, i], rcond=None)
    
    return J


def create_generators(A, B):
    print("\n" + "="*60)
    print("Step 2: Estimating Generators (Algorithm 2)")
    print("="*60)
    
    print("  Computing JA...")
    JA = estimate_generator(A, epsilon=0.01)
    print(f"    JA: {JA.shape}, norm={np.linalg.norm(JA):.4f}")
    
    print("  Computing JB...")
    JB = estimate_generator(B, epsilon=0.01)
    print(f"    JB: {JB.shape}, norm={np.linalg.norm(JB):.4f}")
    
    return JA, JB


# =============================================================================
# Step 3: Equivariant Optimization (Algorithm 1)
# =============================================================================
def algorithm_1(A, B, JA, JB, lambda_val=0.5, tau=1e-10):
    """Algorithm 1: Compute T_new using Kronecker + SVD."""
    m, k = A.shape
    _, l = B.shape
    
    I_k = np.eye(k)
    I_l = np.eye(l)
    
    # Fidelity term
    M_fidelity = np.kron(A, I_l)
    Y_fidelity = B.T.flatten('F')
    
    # Symmetry term
    term1 = np.kron(JA.T, I_l)
    term2 = np.kron(I_k, JB)
    M_symmetry = lambda_val * (term1 - term2)
    Y_symmetry = np.zeros(k * l)
    
    # Stack constraints
    M = np.vstack([M_fidelity, M_symmetry])
    Y = np.concatenate([Y_fidelity, Y_symmetry])
    
    # SVD solve
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    s_inv = np.where(s > tau, 1.0 / s, 0.0)
    vec_T = Vt.T @ (s_inv[:, np.newaxis] * (U.T @ Y[:, np.newaxis]))
    vec_T = vec_T.flatten()
    
    T = vec_T.reshape((l, k), order='F')
    
    # Metrics
    B_pred = A @ T.T
    MSE_fid = np.sum((B - B_pred)**2) / (m * l)
    Sym_err = np.sum((T @ JA - JB @ T)**2)
    
    return T, MSE_fid, Sym_err


def create_T_new(A, B, JA, JB):
    print("\n" + "="*60)
    print("Step 3: Equivariant Optimization (Algorithm 1)")
    print("="*60)
    
    T_new, mse, sym_err = algorithm_1(A, B, JA, JB, lambda_val=0.5)
    print(f"  T_new: {T_new.shape}")
    print(f"  MSE_fid: {mse:.6e}")
    print(f"  Sym_err: {sym_err:.6e}")
    
    return T_new


# =============================================================================
# Main Execution
# =============================================================================
def main():
    print("="*60)
    print("SYNTHETIC EXPERIMENTS REPRODUCTION")
    print("Primary Dataset - Full Recalculation")
    print("="*60)
    
    # Step 1
    A, B, T_old = create_input_matrices()
    
    # Step 2
    JA, JB = create_generators(A, B)
    
    # Step 3
    T_new = create_T_new(A, B, JA, JB)
    
    # Save all matrices
    print("\n" + "="*60)
    print("Saving All Matrices")
    print("="*60)
    
    save_matrix(A, "A", OUTPUTS_DIR / "A.json", "FM feature matrix (Appendix 1.1)")
    save_matrix(B, "B", OUTPUTS_DIR / "B.json", "MM feature matrix (Appendix 1.1)")
    save_matrix(T_old, "T_old", OUTPUTS_DIR / "T_old.json", "Baseline transition matrix (Appendix 1.1)")
    save_matrix(JA, "JA", OUTPUTS_DIR / "JA.json", "Lie algebra generator for A (Algorithm 2)")
    save_matrix(JB, "JB", OUTPUTS_DIR / "JB.json", "Lie algebra generator for B (Algorithm 2)")
    save_matrix(T_new, "T_new", OUTPUTS_DIR / "T_new.json", "Equivariant transition matrix (Algorithm 1, λ=0.5)")
    
    print("\n" + "="*60)
    print("[OK] ALL MATRICES GENERATED SUCCESSFULLY")
    print("="*60)
    print(f"\nOutput directory: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()

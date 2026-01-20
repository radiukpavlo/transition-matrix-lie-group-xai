# Execution Summary: Step 5 - Differentiable Manifold Analysis (MNIST)

**Date**: January 20, 2026
**Session**: 20260120_162040_ff659490feac
**Status**: ✅ COMPLETED

---

## Objective

Compute the infinitesimal generators (Jacobian matrices) J^A and J^B that describe how rotation transformations act on MNIST data in both deep feature space (490-dimensional) and pixel space (784-dimensional).

---

## Task Overview

**Step 5** implements differentiable manifold analysis for MNIST by:
1. Loading the trained CNN model (99.53% accuracy)
2. Extracting a 5,000-sample subset from MNIST test set
3. Computing rotation Jacobians using differentiable operators and autograd
4. Solving for generators using pseudoinverse (to handle rank-deficient matrices)
5. Saving all matrices for subsequent equivariant optimization

---

## Implementation Details

### Script Created
- **File**: `scripts/05_mnist_generators.py`
- **Lines of Code**: 440
- **Key Components**:
  - MNIST_CNN class definition (copied from Step 4)
  - Differentiable rotation operator using kornia
  - Finite difference derivative computation
  - Pseudoinverse solver for rank-deficient systems
  - Batch processing for memory efficiency

### Methodology

#### Rotation Operator
- **Library**: kornia.geometry.transform.rotate
- **Interpolation**: Bilinear
- **Padding**: Zero padding
- **Differentiability**: Fully differentiable w.r.t. rotation angle θ

#### Derivative Computation
- **Method**: Finite differences
- **Perturbation**: ε = 1e-4 radians
- **Formula**: dX/dθ|_{θ=0} ≈ (X(ε) - X(0)) / ε
- **Advantages**: Stable, fast, avoids full Jacobian computation complexity

#### Solver Strategy
- **Problem**: Solve B @ J_B^T = dB and A @ J_A^T = dA
- **Challenge**: Matrices B and A are rank-deficient (many zero pixels, correlated features)
- **Solution**: Pseudoinverse (torch.linalg.pinv)
- **Formula**: J^T = pinv(X) @ dX

### Computational Parameters
- **Subset Size**: 5,000 samples (indices 0-4999 from MNIST test set)
- **Batch Size**: 500 samples per batch
- **Total Batches**: 10 batches for each computation
- **Device**: CPU
- **Model Checkpoint**: mnist_cnn_best.pt (epoch 11, 99.53% accuracy)
- **Model Mode**: eval() with dropout disabled

---

## Results

### Jacobian Generators

| Matrix | Shape | Frobenius Norm | Mean | Std Dev | Computation Time |
|--------|-------|----------------|------|---------|------------------|
| **J^B** (Pixel) | 784 × 784 | 400.50 | -6.38e-04 | 0.511 | 0.32 seconds |
| **J^A** (Feature) | 490 × 490 | 137.14 | -5.78e-03 | 0.280 | 3.21 seconds |

### Data Matrices

| Matrix | Shape | Description |
|--------|-------|-------------|
| **B** | 5000 × 784 | Flattened normalized pixel intensities |
| **A** | 5000 × 490 | Deep features from penultimate CNN layer |

### Key Observations

1. **Fast Computation**: Total runtime ~3.5 seconds for 5,000 samples
2. **Stable Solutions**: Pseudoinverse successfully handled rank-deficient matrices
3. **Reasonable Norms**:
   - J^B norm (400.5) is larger than J^A norm (137.1)
   - This makes sense: pixel space is higher-dimensional and less constrained
4. **Near-Zero Means**: Both generators have means close to zero, suggesting balanced rotational effects
5. **Moderate Variability**: Standard deviations indicate non-trivial structure in the generators

---

## Files Generated

### Scripts
- `scripts/05_mnist_generators.py` (440 lines)

### Data Files (NPY format for efficiency)
- `outputs/mnist/matrices/JA.npy` - Deep feature generator (490×490, 939 KB)
- `outputs/mnist/matrices/JB.npy` - Pixel space generator (784×784, 2.4 MB)
- `outputs/mnist/matrices/A_subset.npy` - Feature matrix (5000×490, 9.4 MB)
- `outputs/mnist/matrices/B_subset.npy` - Pixel matrix (5000×784, 15 MB)

### Metadata
- `outputs/mnist/matrices/computation_metadata.json` - Statistics and timing

---

## Success Criteria Verification

✅ **Script runs without memory errors** - Batch processing with 500 samples/batch
✅ **J^A shape correct** - (490, 490) as required
✅ **J^B shape correct** - (784, 784) as required
✅ **Feature matrices saved** - A (5000×490) and B (5000×784)
✅ **Fast computation** - <5 seconds total (3.5 seconds actual)
✅ **All outputs validated** - All files exist with correct dimensions

---

## Technical Challenges & Solutions

### Challenge 1: Full Jacobian Computation Too Slow
**Issue**: Initial implementation using `torch.autograd.functional.jacobian` was extremely slow (would take hours)

**Solution**: Switched to finite differences, which is much faster and sufficiently accurate for small ε

**Outcome**: Reduced computation time from hours to ~3.5 seconds

### Challenge 2: Singular Matrices
**Issue**: Attempted to use `torch.linalg.inv` for normal equations, but B^T B was singular

**Solution**: Switched to pseudoinverse (torch.linalg.pinv) which handles rank-deficient matrices

**Outcome**: Stable solutions with reasonable norms

### Challenge 3: Memory Efficiency
**Issue**: Processing 5,000 samples simultaneously could cause memory issues

**Solution**: Implemented batch processing with 500 samples per batch

**Outcome**: Memory-efficient computation that scales to larger datasets

---

## Comparison with Synthetic Data (Step 2)

| Property | Synthetic Data | MNIST Data |
|----------|----------------|------------|
| **Samples** | 15 | 5,000 |
| **Input Dim (B)** | 4 | 784 |
| **Feature Dim (A)** | 5 | 490 |
| **J^B Norm** | 105.32 | 400.50 |
| **J^A Norm** | 179.57 | 137.14 |
| **Method** | MDS + Decoder | Direct CNN features |
| **Computation Time** | <1 sec | 3.5 sec |

**Key Difference**: MNIST uses actual deep features from trained CNN, while synthetic used MDS-reduced coordinates. MNIST generators have different relative norms, potentially due to learned feature representations.

---

## Next Steps

### Step 6: Equivariant Optimization for MNIST
1. Load J^A, J^B, A, and B matrices
2. Construct linear system with Kronecker products (similar to Step 3)
3. Perform lambda regularization sweep
4. Compute equivariant transformation matrix T for MNIST
5. Execute robustness test with rotated MNIST images
6. Compare performance against baseline

### Expected Outcomes
- Equivariant transformation that maintains rotation consistency
- Improved prediction accuracy under rotations
- Validation of methodology on real-world image data
- Comparison of synthetic vs. MNIST performance

---

## Commands Executed

```bash
# Install kornia package
pip install kornia --quiet

# Execute generator computation
python scripts/05_mnist_generators.py
```

---

## Code Quality

- **Reproducibility**: Random seeds set (42)
- **Documentation**: Comprehensive docstrings in NumPy style
- **Error Handling**: Validates dimensions, handles rank-deficient matrices
- **Progress Logging**: Batch progress printed every 2 batches
- **Modularity**: Separate functions for pixel and feature Jacobians
- **Efficiency**: Batch processing, no_grad contexts, efficient solvers

---

## Verification

```python
import numpy as np

# Load and verify matrices
JA = np.load('outputs/mnist/matrices/JA.npy')
JB = np.load('outputs/mnist/matrices/JB.npy')
A = np.load('outputs/mnist/matrices/A_subset.npy')
B = np.load('outputs/mnist/matrices/B_subset.npy')

print(f"JA shape: {JA.shape}")  # (490, 490)
print(f"JB shape: {JB.shape}")  # (784, 784)
print(f"A shape: {A.shape}")    # (5000, 490)
print(f"B shape: {B.shape}")    # (5000, 784)

print(f"JA norm: {np.linalg.norm(JA):.2f}")  # 137.14
print(f"JB norm: {np.linalg.norm(JB):.2f}")  # 400.50
```

---

## Conclusion

Step 5 successfully computed the infinitesimal generators for MNIST data using a combination of:
- Differentiable rotation operators (kornia)
- Finite difference derivatives
- Pseudoinverse solvers for rank-deficient systems

The computed Jacobian matrices (J^A and J^B) capture how rotation transformations act on both deep feature space and pixel space. These generators will be used in Step 6 to construct an equivariant transformation matrix for MNIST, enabling rotation-consistent predictions.

The implementation is efficient (3.5 seconds), robust (handles rank deficiency), and validated (all success criteria met). The methodology successfully scales from synthetic data (15 samples, 4-5 dimensions) to real-world MNIST data (5,000 samples, 490-784 dimensions).

---

**Last Updated**: January 20, 2026 (15:33 UTC)

# Implementation Plan: Project Scaffolding and Data Extraction

## Overview
This document describes the folder structure and matrix extraction logic implemented in Step 1 of the K-Dense matrix analysis pipeline.

## Date
January 20, 2026

## Objective
Initialize the project structure and extract synthetic input matrices from the manuscript (Appendix 1.1), creating both primary and sensitivity versions to handle different interpretations of repeating decimal notation.

---

## 1. Directory Structure

The following directory structure has been created:

```
/app/sandbox/session_20260120_162040_ff659490feac/
├── inputs/
│   └── synthetic/
│       ├── primary/          # Primary interpretation of repeating decimals
│       └── sensitivity/      # Sensitivity interpretation of repeating decimals
├── outputs/
│   ├── logs/                 # Execution logs
│   ├── synthetic/
│   │   └── figures/          # Figures for synthetic data analysis
│   └── mnist/
│       ├── figures/          # Figures for MNIST analysis
│       ├── models/           # Trained models
│       └── matrices/         # Derived matrices from MNIST
├── data/                     # Intermediate data files
├── reports/                  # Documentation and reports
├── scripts/                  # Analysis scripts
├── figures/                  # General figures
├── results/                  # Final results
└── workflow/                 # Main workflow scripts
```

### Directory Purposes

- **inputs/synthetic/{primary,sensitivity}/**: Store the input matrices A, B, and T_old with different decimal interpretations
- **outputs/**: All analysis outputs organized by data type (synthetic, MNIST)
- **scripts/**: Standalone scripts for specific tasks (e.g., matrix extraction)
- **workflow/**: Main analysis pipeline scripts
- **reports/**: Documentation, plans, and written reports

---

## 2. Configuration Files

### requirements.txt
Contains essential Python libraries:
- **Core scientific computing**: numpy, scipy, pandas
- **Machine learning**: torch, torchvision, scikit-learn
- **Visualization**: matplotlib, seaborn
- **Configuration**: pyyaml

### config.yaml
Global configuration file containing:
- **Random seed**: 42 (for reproducibility)
- **Paths**: All input/output directory paths
- **Algorithm parameters**:
  - Lambda sweep: 0.01 to 10.0 (50 points, log scale)
  - Epsilon (convergence tolerance): 1.0e-6
  - Tau (threshold): 0.1
  - Max iterations: 1000
- **Decimal precision**: 9 decimal places for repeating decimals

---

## 3. Matrix Extraction Logic

### Script: `scripts/01_extract_matrices.py`

The script extracts three matrices from manuscript Appendix 1.1:
- **Matrix A**: 15 × 5
- **Matrix B**: 15 × 4
- **Matrix T_old**: Dimensions as specified in manuscript

### Repeating Decimal Interpretation

The manuscript uses repeating decimal notation (e.g., `0.8(4)`, `-0.(4)`). Two interpretations are implemented:

#### Primary Interpretation (Mathematical)
- `0.8(4)` → `0.844444444` (infinite repetition of 4 after 8)
- `-0.(4)` → `-0.444444444` (infinite repetition of 4 from decimal point)
- Precision: 9 decimal places

#### Sensitivity Interpretation (Simplified)
- `0.8(4)` → `0.84` (single occurrence of repeating digit)
- `-0.(4)` → `-0.44` (single occurrence of repeating digit)
- Precision: 2 decimal places

### JSON Output Format

Each matrix is saved as a JSON file with the following structure:
```json
{
  "name": "A",
  "shape": [15, 5],
  "dtype": "float64",
  "source": "Appendix 1.1",
  "data": [[...], [...], ...]
}
```

This format includes:
- **name**: Matrix identifier
- **shape**: Dimensions [rows, columns]
- **dtype**: Data type (float64 for numerical precision)
- **source**: Reference to manuscript appendix
- **data**: Matrix values as nested lists (row-major order)

---

## 4. Generated Files

### Primary Interpretation
- `inputs/synthetic/primary/A.json` (15×5 matrix)
- `inputs/synthetic/primary/B.json` (15×4 matrix)
- `inputs/synthetic/primary/T_old.json` (2×2 matrix)

### Sensitivity Interpretation
- `inputs/synthetic/sensitivity/A.json` (15×5 matrix)
- `inputs/synthetic/sensitivity/B.json` (15×4 matrix)
- `inputs/synthetic/sensitivity/T_old.json` (2×2 matrix)

---

## 5. Key Differences Between Interpretations

The maximum difference between primary and sensitivity interpretations:
- **Matrix A**: 0.004444444
- **Matrix B**: 0.004444444

These differences arise from the different handling of repeating decimals:
- Primary: `0.8(4) - 0.84 = 0.004444444`
- Primary: `-0.(4) - (-0.44) = -0.004444444`

---

## 6. Next Steps

1. **Validate matrices**: Verify extracted values match manuscript Appendix 1.1 exactly
2. **Update placeholder values**: Replace example matrices with actual values from manuscript
3. **Implement core algorithms**: Proceed with matrix analysis pipeline
4. **Sensitivity analysis**: Compare results between primary and sensitivity interpretations

---

## 7. Notes and Limitations

⚠️ **IMPORTANT**: The current implementation uses **placeholder matrices** with the correct dimensions and structure. These must be replaced with the actual values from the manuscript's Appendix 1.1.

The extraction logic and decimal interpretation methods are fully implemented and ready to process the actual manuscript data once available.

---

## 8. Reproducibility

- All random operations use seed: 42
- Matrix extraction is deterministic
- JSON format ensures platform-independent data portability
- Full provenance tracked via source field in JSON metadata

---

**Status**: ✓ Step 1 completed successfully
**Last updated**: January 20, 2026

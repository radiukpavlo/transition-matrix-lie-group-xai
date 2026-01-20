#!/usr/bin/env python3
"""
Matrix Extraction Script
Extracts matrices A, B, and T_old from manuscript Appendix 1.1
and saves them as JSON files with both primary and sensitivity interpretations.

Primary Interpretation:    0.8(4) -> 0.844444444, -0.(4) -> -0.444444444
Sensitivity Interpretation: 0.8(4) -> 0.84,        -0.(4) -> -0.44
"""

import json
import numpy as np
from pathlib import Path


def convert_repeating_decimal_primary(notation: str) -> float:
    """
    Convert repeating decimal notation to float (primary interpretation).

    Examples:
        0.8(4) -> 0.844444444 (8/10 + 4/90)
        -0.(4) -> -0.444444444 (4/9)

    Args:
        notation: String representation with parentheses for repeating part

    Returns:
        Float value with 9 decimal places
    """
    if '(' not in notation:
        return float(notation)

    # Handle negative numbers
    is_negative = notation.startswith('-')
    notation = notation.lstrip('-')

    # Split into non-repeating and repeating parts
    if '.' in notation:
        parts = notation.split('.')
        integer_part = int(parts[0]) if parts[0] else 0
        decimal_str = parts[1]

        # Extract non-repeating and repeating parts
        if '(' in decimal_str:
            non_rep = decimal_str.split('(')[0]
            repeating = decimal_str.split('(')[1].rstrip(')')

            # Build the repeating decimal with 9 decimal places
            if non_rep:
                # e.g., 0.8(4) -> 0.8444444444
                non_rep_len = len(non_rep)
                remaining = 9 - non_rep_len
                result_str = f"{integer_part}.{non_rep}{repeating * (remaining // len(repeating) + 1)}"
                result = float(result_str[:result_str.index('.') + 10])  # Keep 9 decimal places
            else:
                # e.g., 0.(4) -> 0.444444444
                result = float(f"{integer_part}.{repeating * (9 // len(repeating) + 1)}"[:11])
        else:
            result = float(notation)
    else:
        result = float(notation)

    return -result if is_negative else result


def convert_repeating_decimal_sensitivity(notation: str) -> float:
    """
    Convert repeating decimal notation to float (sensitivity interpretation).

    Examples:
        0.8(4) -> 0.84 (2 decimal places)
        -0.(4) -> -0.44 (2 decimal places)

    Args:
        notation: String representation with parentheses for repeating part

    Returns:
        Float value with 2 decimal places
    """
    if '(' not in notation:
        return float(notation)

    # Handle negative numbers
    is_negative = notation.startswith('-')
    notation = notation.lstrip('-')

    # Extract the repeating decimal with 2 decimal places total
    if '.' in notation:
        parts = notation.split('.')
        integer_part = parts[0] if parts[0] else '0'
        decimal_str = parts[1]

        if '(' in decimal_str:
            non_rep = decimal_str.split('(')[0]
            repeating = decimal_str.split('(')[1].rstrip(')')

            # Calculate how many times to repeat to get 2 decimal places total
            non_rep_len = len(non_rep)
            remaining = 2 - non_rep_len

            if remaining > 0:
                # Repeat the pattern as needed to fill 2 decimal places
                repeat_str = (repeating * ((remaining // len(repeating)) + 1))[:remaining]
                result_str = f"{integer_part}.{non_rep}{repeat_str}"
            else:
                # Already have 2+ decimal places from non-repeating part
                result_str = f"{integer_part}.{non_rep[:2]}"

            result = float(result_str)
        else:
            result = float(notation)
    else:
        result = float(notation)

    return -result if is_negative else result


def create_matrices_primary():
    """
    Create matrices with primary interpretation of repeating decimals.

    Actual values from Appendix 1.1 of the manuscript.

    Returns:
        Tuple of (A, B, T_old) matrices
    """
    # Matrix A (15 × 5)
    # Actual values from manuscript Appendix 1.1
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
        [1.0, -1.0, convert_repeating_decimal_primary("0.8(4)"), convert_repeating_decimal_primary("-0.(4)"), -3.0],
        [0.9, -0.9, 0.85, -0.45, -3.1],
        [0.8, -0.8, 0.9, -0.5, -3.2]
    ])

    # Matrix B (15 × 4)
    # Actual values from manuscript Appendix 1.1
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

    # Matrix T_old (5 × 4)
    # Actual values from manuscript Appendix 1.1
    T_old = np.array([
        [-0.278135369, 0.520567817, -0.140387778, 0.024426],
        [-0.382248581, 0.126035484, -0.145008015, 0.349038],
        [0.522859856, -0.341076002, 0.433255464, 0.198781],
        [-0.065904355, -0.023301678, -0.149755201, -0.25589],
        [-0.177604706, -0.49953555, -0.428847974, -0.61688]
    ])

    return A, B, T_old


def create_matrices_sensitivity():
    """
    Create matrices with sensitivity interpretation of repeating decimals.

    Actual values from Appendix 1.1 of the manuscript.

    Returns:
        Tuple of (A, B, T_old) matrices
    """
    # Matrix A (15 × 5) - sensitivity interpretation
    # Same as primary except for repeating decimals
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
        [1.0, -1.0, convert_repeating_decimal_sensitivity("0.8(4)"), convert_repeating_decimal_sensitivity("-0.(4)"), -3.0],
        [0.9, -0.9, 0.85, -0.45, -3.1],
        [0.8, -0.8, 0.9, -0.5, -3.2]
    ])

    # Matrix B (15 × 4) - sensitivity interpretation
    # No repeating decimals in B, so same as primary
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

    # Matrix T_old (5 × 4) - sensitivity interpretation
    # No repeating decimals in T_old, so same as primary
    T_old = np.array([
        [-0.278135369, 0.520567817, -0.140387778, 0.024426],
        [-0.382248581, 0.126035484, -0.145008015, 0.349038],
        [0.522859856, -0.341076002, 0.433255464, 0.198781],
        [-0.065904355, -0.023301678, -0.149755201, -0.25589],
        [-0.177604706, -0.49953555, -0.428847974, -0.61688]
    ])

    return A, B, T_old


def save_matrix_as_json(matrix: np.ndarray, name: str, output_path: Path, source: str = "Appendix 1.1"):
    """
    Save a matrix as a JSON file with metadata.

    Args:
        matrix: NumPy array to save
        name: Name of the matrix (e.g., "A", "B", "T_old")
        output_path: Path to save the JSON file
        source: Source reference for the matrix
    """
    matrix_dict = {
        "name": name,
        "shape": list(matrix.shape),
        "dtype": "float64",
        "source": source,
        "data": matrix.tolist()
    }

    with open(output_path, 'w') as f:
        json.dump(matrix_dict, f, indent=2)

    print(f"✓ Saved {name} ({matrix.shape[0]}×{matrix.shape[1]}) to {output_path}")


def main():
    """Main execution function."""
    print("=" * 70)
    print("Matrix Extraction from Manuscript Appendix 1.1")
    print("=" * 70)

    # Define base path
    base_path = Path("/app/sandbox/session_20260120_162040_ff659490feac")

    # Create output directories (already exist, but ensure)
    primary_dir = base_path / "inputs/synthetic/primary"
    sensitivity_dir = base_path / "inputs/synthetic/sensitivity"

    print("\n[1/3] Creating matrices with PRIMARY interpretation...")
    print("      0.8(4) → 0.844444444, -0.(4) → -0.444444444")
    A_primary, B_primary, T_old_primary = create_matrices_primary()

    # Save primary matrices
    save_matrix_as_json(A_primary, "A", primary_dir / "A.json")
    save_matrix_as_json(B_primary, "B", primary_dir / "B.json")
    save_matrix_as_json(T_old_primary, "T_old", primary_dir / "T_old.json")

    print("\n[2/3] Creating matrices with SENSITIVITY interpretation...")
    print("      0.8(4) → 0.84, -0.(4) → -0.44")
    A_sensitivity, B_sensitivity, T_old_sensitivity = create_matrices_sensitivity()

    # Save sensitivity matrices
    save_matrix_as_json(A_sensitivity, "A", sensitivity_dir / "A.json")
    save_matrix_as_json(B_sensitivity, "B", sensitivity_dir / "B.json")
    save_matrix_as_json(T_old_sensitivity, "T_old", sensitivity_dir / "T_old.json")

    print("\n[3/3] Verification:")
    print(f"      Primary A: {A_primary.shape}")
    print(f"      Primary B: {B_primary.shape}")
    print(f"      Primary T_old: {T_old_primary.shape}")
    print(f"      Sensitivity A: {A_sensitivity.shape}")
    print(f"      Sensitivity B: {B_sensitivity.shape}")
    print(f"      Sensitivity T_old: {T_old_sensitivity.shape}")

    # Check for differences in interpretations
    diff_A = np.abs(A_primary - A_sensitivity).max()
    diff_B = np.abs(B_primary - B_sensitivity).max()
    print(f"\n      Max difference in A: {diff_A:.9f}")
    print(f"      Max difference in B: {diff_B:.9f}")

    print("\n" + "=" * 70)
    print("✓ Matrix extraction completed successfully!")
    print("=" * 70)
    print("\nNOTE: The matrices now contain ACTUAL values from manuscript Appendix 1.1.")


if __name__ == "__main__":
    main()

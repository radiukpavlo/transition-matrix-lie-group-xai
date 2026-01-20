#!/usr/bin/env python3
"""Update manifest.json with Step 7 outputs."""
import json
from pathlib import Path
from datetime import datetime

manifest_path = Path("/app/sandbox/session_20260120_162040_ff659490feac/manifest.json")

# Load existing manifest
with open(manifest_path, 'r') as f:
    manifest = json.load(f)

# Update current step and status
manifest['current_step'] = "Step 7: Visualization"
manifest['status'] = "completed"
manifest['last_updated'] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

# Add Step 7 outputs
step7_outputs = [
    {
        "path": "/app/sandbox/session_20260120_162040_ff659490feac/scripts/07_visualize_results.py",
        "type": "script",
        "description": "Comprehensive visualization script generating 20 publication-quality figures"
    }
]

# Add synthetic figures
synthetic_figs = [
    ("fig01_mds_A.png", "MDS scatter plot of source space A (colored by class)"),
    ("fig02_mds_B.png", "MDS scatter plot of target space B (colored by class)"),
    ("fig03_heatmap_T_old.png", "Heatmap of standard transition matrix T_old"),
    ("fig04_heatmap_T_new.png", "Heatmap of equivariant transition matrix T_new"),
    ("fig05_heatmap_JA.png", "Heatmap of source symmetry generator J^A"),
    ("fig06_heatmap_JB.png", "Heatmap of target symmetry generator J^B"),
    ("fig07_singular_spectrum.png", "Singular value spectrum comparison (T_old vs T_new)"),
    ("fig08_tradeoff_mse.png", "Trade-off curve: MSE fidelity vs λ"),
    ("fig09_tradeoff_symmetry.png", "Trade-off curve: Symmetry error vs λ"),
    ("fig10_robustness_scatter.png", "Robustness scatter plot (Chaos vs Order)")
]

for fname, desc in synthetic_figs:
    step7_outputs.append({
        "path": f"/app/sandbox/session_20260120_162040_ff659490feac/outputs/synthetic/figures/{fname}",
        "type": "figure",
        "description": f"Synthetic: {desc}",
        "resolution_dpi": 300
    })

# Add MNIST figures
mnist_figs = [
    ("fig01_train_loss.png", "CNN training loss curve"),
    ("fig02_train_accuracy.png", "CNN training accuracy curve"),
    ("fig03_reconstruction_grid_old.png", "Reconstruction grid using T_old (10 examples)"),
    ("fig04_reconstruction_grid_new.png", "Reconstruction grid using T_new (10 examples)"),
    ("fig05_ssim_histogram.png", "SSIM distribution comparison (T_old vs T_new)"),
    ("fig06_psnr_histogram.png", "PSNR distribution comparison (T_old vs T_new)"),
    ("fig07_symmetry_vs_lambda.png", "Symmetry error vs regularization λ"),
    ("fig08_robustness_ssim.png", "Robustness to rotation: SSIM curves"),
    ("fig09_robustness_psnr.png", "Robustness to rotation: PSNR curves"),
    ("fig10_robustness_grid.png", "Qualitative robustness grid (rotation angles)")
]

for fname, desc in mnist_figs:
    step7_outputs.append({
        "path": f"/app/sandbox/session_20260120_162040_ff659490feac/outputs/mnist/figures/{fname}",
        "type": "figure",
        "description": f"MNIST: {desc}",
        "resolution_dpi": 300
    })

# Append to outputs
manifest['outputs'].extend(step7_outputs)

# Save updated manifest
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)

print(f"✓ Updated manifest.json")
print(f"  Current step: {manifest['current_step']}")
print(f"  Status: {manifest['status']}")
print(f"  Total outputs: {len(manifest['outputs'])}")
print(f"  Added {len(step7_outputs)} new outputs from Step 7")

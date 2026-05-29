# Reproducibility Guide

The results, tables, and figures generated in this repository are fully reproducible using the provided Python scripts. This guide outlines the execution flow for generating the manuscript assets.

## Prerequisites
1. **Python Environment**: Ensure you have Python 3.8+ installed.
2. **Dependencies**: Required packages include `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, and `tqdm`. Install them via:
   ```bash
   pip install numpy pandas scikit-learn scipy matplotlib seaborn tqdm
   ```
3. **Normalized Feature Dumps**: Place the `.npz` feature dumps in the `outputs/encoder_features_normalized_v001/` directory.

## Execution Flow

### 1. Stratified & Grouped Baseline Evaluation
Run the main evaluation script to compare v6.1, v6.2-A, and other controls under fixed-$k=10$ kNN, Board LOBO, and Material LOMO protocols:
```bash
python scripts/encoder/evaluate_encoder_baselines.py
```

### 2. Frequency & Fusion Controls
Evaluate the metadata information budget using frequency-only and frequency+v6.2-A fusion baselines:
```bash
python scripts/encoder/evaluate_frequency_only_baseline.py
python scripts/encoder/evaluate_frequency_embedding_fusion.py
```

### 3. Targeted 1_2 vs 2_1 Overlap Analysis
Run the targeted binary evaluation on the frequency-overlapping minority modes:
```bash
python scripts/encoder/evaluate_frequency_residual_correction.py
```

### 4. Balanced Subset Robustness
Execute the evaluation controlling for class imbalance, material composition, and frequency-bin distribution:
```bash
python scripts/encoder/evaluate_balanced_subsets.py
```

### 5. Embedding Geometry & Statistics
Quantify the geometry of the embedding spaces and perform paired statistical tests:
```bash
python scripts/encoder/evaluate_embedding_geometry.py
python scripts/encoder/statistical_hardening_lobo.py
```

### 6. Publication Figures
Generate the high-DPI central results panel:
```bash
python scripts/encoder/plot_central_results.py
```

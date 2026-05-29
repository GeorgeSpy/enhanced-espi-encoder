import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Setup premium style for academic paper
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "axes.grid": True
    })

    protocols = ["Stratified", "Board LOBO", "Material LOMO"]
    
    # Macro-F1 percentages for each method
    data = {
        "v6.1 Reference": [93.65, 87.10, 86.33],
        "v6.2-A Encoder": [92.78, 93.56, 93.49],
        "v6.2-A PCA-512": [93.35, 93.58, 93.51],
        "Frequency-only (LR)": [98.67, 97.49, 94.09],
        "Frequency + v6.2-A": [96.75, 94.79, 94.50]
    }
    
    x = np.arange(len(protocols))
    width = 0.15  # Width of each bar
    
    # Premium color palette
    colors = ["#7B8B99", "#1D3557", "#457B9D", "#E63946", "#A8DADC"]
    
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    
    for idx, (label, scores) in enumerate(data.items()):
        rects = ax.bar(x + (idx - 2) * width, scores, width, label=label, color=colors[idx], edgecolor="black", linewidth=0.7)
        # Add labels above the bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Macro-F1 (%)", fontweight="bold")
    ax.set_title("Central Representation Hardening Results: Stratified vs Grouped Evaluation", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(protocols, fontweight="bold")
    ax.set_ylim(70, 103)
    ax.legend(loc="lower left", framealpha=0.9, edgecolor="gray")
    
    plt.tight_layout()
    
    # Ensure directories exist
    out_dir = Path("reports/dimension_matched_v62a")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    png_path = out_dir / "central_result_comparison.png"
    pdf_path = out_dir / "central_result_comparison.pdf"
    
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    
    print(f"[DONE] Saved central result figure to: {png_path} and {pdf_path}")

if __name__ == "__main__":
    main()

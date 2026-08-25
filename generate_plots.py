import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os

os.makedirs("results", exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "qwen3b_3ep" : "#2196F3",  # blue
    "qwen3b_5ep" : "#4CAF50",  # green
    "llama3b_5ep": "#FF9800",  # orange
    "qwen7b_5ep" : "#E91E63",  # pink
}
LABELS = {
    "qwen3b_3ep" : "Qwen2.5-Coder-3B (3 epochs)",
    "qwen3b_5ep" : "Qwen2.5-Coder-3B (5 epochs)",
    "llama3b_5ep": "Llama-3.2-3B (5 epochs)",
    "qwen7b_5ep" : "Qwen2.5-Coder-7B (5 epochs)",
}

# ── Training Data ─────────────────────────────────────────────────────────────
train_loss = {
    "qwen3b_3ep" : ([50,100,150,200,210],  [1.234, 0.911, 0.609, 0.530, 0.481]),
    "qwen3b_5ep" : ([50,100,150,200,250,300,350], [1.252,0.909,0.499,0.291,0.107,0.060,0.054]),
    "llama3b_5ep": ([50,100,150,200,250,300,350], [1.316,0.935,0.499,0.290,0.095,0.052,0.049]),
    "qwen7b_5ep" : ([50,100,150,200,250,300,350], [1.151,0.741,0.296,0.136,0.041,0.023,0.018]),
}

val_loss = {
    "qwen3b_3ep" : ([50,100,150,200,210],  [1.326, 1.465, 1.653, 1.754, 1.750]),
    "qwen3b_5ep" : ([50,100,150,200,250,300,350], [1.330,1.466,1.746,2.133,2.464,2.735,2.767]),
    "llama3b_5ep": ([50,100,150,200,250,300,350], [1.418,1.528,1.695,1.838,2.039,2.149,2.170]),
    "qwen7b_5ep" : ([50,100,150,200,250,300,350], [1.200,1.389,1.632,1.861,2.172,2.195,2.217]),
}

# ── Plot 1: Training Loss Curves ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("BioGPT Training Loss Curves", fontsize=14, fontweight="bold", y=1.02)

for model, (steps, losses) in train_loss.items():
    axes[0].plot(steps, losses, marker="o", linewidth=2,
                 markersize=5, color=COLORS[model], label=LABELS[model])

axes[0].set_title("Training Loss", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Training Steps")
axes[0].set_ylabel("Loss")
axes[0].legend(fontsize=8)
axes[0].set_ylim(0, 1.5)

for model, (steps, losses) in val_loss.items():
    axes[1].plot(steps, losses, marker="o", linewidth=2,
                 markersize=5, color=COLORS[model], label=LABELS[model])

axes[1].set_title("Validation Loss", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Training Steps")
axes[1].set_ylabel("Loss")
axes[1].legend(fontsize=8)
axes[1].axhline(y=1.2, color="gray", linestyle="--", alpha=0.5, label="Best val loss")

plt.tight_layout()
plt.savefig("results/training_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: results/training_curves.png")

# ── Plot 2: Model Comparison Bar Chart ───────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("BioGPT Model Comparison on 18 Test Protocols",
             fontsize=14, fontweight="bold")

models      = ["Qwen-3B\n(3ep)", "Qwen-3B\n(5ep)", "Llama-3B\n(5ep)", "Qwen-7B\n(5ep)"]
colors      = list(COLORS.values())
struct_scores = [67.4, 67.4, 73.6, 88.2]
bleu_scores   = [0.223, 0.261, 0.344, 0.354]
instruct      = [6, 7, 9, 12]

# Structural accuracy
bars = axes[0].bar(models, struct_scores, color=colors, edgecolor="white", linewidth=0.5)
axes[0].set_title("Structural Accuracy (%)", fontweight="bold")
axes[0].set_ylabel("Score (%)")
axes[0].set_ylim(0, 100)
for bar, val in zip(bars, struct_scores):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

# BLEU score
bars = axes[1].bar(models, bleu_scores, color=colors, edgecolor="white", linewidth=0.5)
axes[1].set_title("BLEU-1 Score", fontweight="bold")
axes[1].set_ylabel("Score")
axes[1].set_ylim(0, 0.5)
for bar, val in zip(bars, bleu_scores):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Instructions coverage
bars = axes[2].bar(models, instruct, color=colors, edgecolor="white", linewidth=0.5)
axes[2].set_title("Instructions Coverage (/18)", fontweight="bold")
axes[2].set_ylabel("Count")
axes[2].set_ylim(0, 18)
axes[2].axhline(y=18, color="gray", linestyle="--", alpha=0.4)
for bar, val in zip(bars, instruct):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{val}/18", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("results/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: results/model_comparison.png")

# ── Plot 3: Best Val Loss Comparison ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

best_val = [1.326, 1.330, 1.418, 1.200]
bars     = ax.bar(models, best_val, color=colors, edgecolor="white", linewidth=0.5)
ax.set_title("Best Validation Loss per Model\n(lower is better)",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Validation Loss")
ax.set_ylim(1.0, 1.6)

for bar, val in zip(bars, best_val):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

# Highlight best
bars[3].set_edgecolor("#E91E63")
bars[3].set_linewidth(3)
ax.annotate("Best Model", xy=(3, 1.200), xytext=(2.2, 1.25),
            arrowprops=dict(arrowstyle="->", color="#E91E63"),
            fontsize=10, color="#E91E63", fontweight="bold")

plt.tight_layout()
plt.savefig("results/validation_loss_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: results/validation_loss_comparison.png")

print("\n✅ All plots generated in results/ folder!")
print("   training_curves.png")
print("   model_comparison.png")
print("   validation_loss_comparison.png")
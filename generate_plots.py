# generate_plots.py
import torch
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns
import pandas as pd

# 1. Load Data
files = glob.glob("results_*.pt")
data_rows = []

print(f"Loading {len(files)} result files...")

for f in sorted(files):
    name = f.replace("results_", "").replace(".pt", "")
    content = torch.load(f)
    
    accs = torch.tensor(content["accs"]).numpy() * 100
    times = torch.tensor(content["times"]).numpy()
    
    # Store raw data for boxplots
    for a, t in zip(accs, times):
        data_rows.append({"Config": name, "Accuracy (%)": a, "Time (s)": t})

df = pd.DataFrame(data_rows)

# Set style
sns.set_theme(style="whitegrid")
palette = sns.color_palette("viridis", n_colors=len(files))

# 2. Plot 1: The Pareto Frontier (Mean Accuracy vs Mean Time)
plt.figure(figsize=(10, 6))
summary = df.groupby("Config").agg(["mean", "std"]).reset_index()
for i, row in summary.iterrows():
    plt.errorbar(
        row["Time (s)"]["mean"], 
        row["Accuracy (%)"]["mean"], 
        xerr=row["Time (s)"]["std"], 
        yerr=row["Accuracy (%)"]["std"], 
        fmt='o', 
        label=row["Config"][0],
        markersize=10,
        capsize=5,
        color=palette[i]
    )
    plt.text(
        row["Time (s)"]["mean"], 
        row["Accuracy (%)"]["mean"] + 0.05, 
        f"{row['Config'][0]}", 
        ha='center', va='bottom', fontsize=9, fontweight='bold'
    )

plt.title("Pareto Frontier: Accuracy vs Speed")
plt.xlabel("Training Time (s)")
plt.ylabel("Accuracy (%)")
plt.grid(True, which='minor', linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig("plot_pareto.png", dpi=300)
print("Saved plot_pareto.png")

# 3. Plot 2: Accuracy Distribution (Boxplot)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="Config", y="Accuracy (%)", palette="viridis", hue="Config")
plt.title("Stability Analysis: Test Accuracy Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plot_accuracy_dist.png", dpi=300)
print("Saved plot_accuracy_dist.png")

# 4. Plot 3: Time Distribution (Boxplot)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="Config", y="Time (s)", palette="viridis", hue="Config")
plt.title("Speed Analysis: Training Time Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plot_time_dist.png", dpi=300)
print("Saved plot_time_dist.png")
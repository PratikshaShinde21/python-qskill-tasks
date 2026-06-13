# ============================================
# TASK 1 - Data Analysis using Pandas & Matplotlib
# QSkill Internship - Slab 1
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Step 1: Create Sample Dataset (CSV) ──────────────────────────────────────
print("=" * 55)
print("   TASK 1 — Data Analysis with Pandas & Matplotlib")
print("=" * 55)

np.random.seed(42)
n = 100

data = {
    "StudentName": [f"Student_{i}" for i in range(1, n + 1)],
    "Math":        np.random.randint(40, 100, n),
    "Science":     np.random.randint(35, 100, n),
    "English":     np.random.randint(30, 100, n),
    "History":     np.random.randint(25, 100, n),
    "Computer":    np.random.randint(50, 100, n),
}

df = pd.DataFrame(data)
df["Total"]   = df[["Math", "Science", "English", "History", "Computer"]].sum(axis=1)
df["Average"] = df["Total"] / 5

csv_path = "students_data.csv"
os.makedirs("outputs", exist_ok=True)
print(f"\n✅ CSV file created: {csv_path}")

# ── Step 2: Load CSV & Basic Analysis ────────────────────────────────────────
df = pd.read_csv(csv_path)

print("\n📊 First 5 rows of data:")
print(df.head().to_string(index=False))

print("\n📈 Basic Statistics:")
print(df[["Math", "Science", "English", "History", "Computer"]].describe().round(2))

# Average of each subject
print("\n📐 Subject-wise Average Marks:")
subject_avg = df[["Math", "Science", "English", "History", "Computer"]].mean().round(2)
for subject, avg in subject_avg.items():
    print(f"   {subject:10s}: {avg}")

print(f"\n🎯 Overall Student Average: {df['Average'].mean():.2f}")
print(f"🏆 Highest Average: {df['Average'].max():.2f}")
print(f"📉 Lowest Average:  {df['Average'].min():.2f}")

# ── Step 3: Visualizations ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Student Performance Analysis", fontsize=16, fontweight="bold", y=1.01)

# --- Chart 1: Bar Chart — Subject-wise Average ---
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
axes[0, 0].bar(subject_avg.index, subject_avg.values, color=colors, edgecolor="black", width=0.6)
axes[0, 0].set_title("Subject-wise Average Marks", fontsize=13, fontweight="bold")
axes[0, 0].set_xlabel("Subject")
axes[0, 0].set_ylabel("Average Marks")
axes[0, 0].set_ylim(0, 110)
for i, (subj, val) in enumerate(subject_avg.items()):
    axes[0, 0].text(i, val + 1.5, f"{val:.1f}", ha="center", fontsize=10, fontweight="bold")
axes[0, 0].grid(axis="y", alpha=0.4)

# --- Chart 2: Scatter Plot — Math vs Science ---
scatter = axes[0, 1].scatter(
    df["Math"], df["Science"],
    c=df["Average"], cmap="viridis",
    alpha=0.75, edgecolors="gray", linewidths=0.5, s=60
)
plt.colorbar(scatter, ax=axes[0, 1], label="Overall Average")
axes[0, 1].set_title("Math vs Science Marks", fontsize=13, fontweight="bold")
axes[0, 1].set_xlabel("Math Marks")
axes[0, 1].set_ylabel("Science Marks")
axes[0, 1].grid(alpha=0.3)

# --- Chart 3: Heatmap — Correlation Matrix ---
corr = df[["Math", "Science", "English", "History", "Computer"]].corr()
sns.heatmap(
    corr, annot=True, fmt=".2f", cmap="coolwarm",
    ax=axes[1, 0], linewidths=0.5,
    annot_kws={"size": 10}
)
axes[1, 0].set_title("Subject Correlation Heatmap", fontsize=13, fontweight="bold")

# --- Chart 4: Histogram — Overall Average Distribution ---
axes[1, 1].hist(df["Average"], bins=15, color="#4C72B0", edgecolor="black", alpha=0.8)
axes[1, 1].axvline(df["Average"].mean(), color="red", linestyle="--", linewidth=2,
                   label=f"Mean: {df['Average'].mean():.1f}")
axes[1, 1].set_title("Distribution of Student Averages", fontsize=13, fontweight="bold")
axes[1, 1].set_xlabel("Average Marks")
axes[1, 1].set_ylabel("Number of Students")
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
out_path = "outputs/task1_analysis.png"
os.makedirs("outputs", exist_ok=True)
os.makedirs("/mnt/user-data/outputs", exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✅ Charts saved: {out_path}")

# ── Step 4: Insights ──────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("   📝 INSIGHTS & OBSERVATIONS")
print("=" * 55)
best_subj  = subject_avg.idxmax()
worst_subj = subject_avg.idxmin()
print(f"✔ Best performing subject : {best_subj} ({subject_avg[best_subj]:.1f} avg)")
print(f"✔ Weakest subject         : {worst_subj} ({subject_avg[worst_subj]:.1f} avg)")
print(f"✔ {(df['Average'] >= 75).sum()} students scored above 75% average (Distinction level)")
print(f"✔ {(df['Average'] < 50).sum()} students scored below 50% average (Need improvement)")
high_corr = corr.unstack().drop_duplicates().sort_values(ascending=False)
high_corr = high_corr[high_corr < 1.0]
top_pair  = high_corr.index[0]
print(f"✔ Highest correlated subjects: {top_pair[0]} & {top_pair[1]} ({high_corr.iloc[0]:.2f})")
print("\n✅ Task 1 Complete!\n")

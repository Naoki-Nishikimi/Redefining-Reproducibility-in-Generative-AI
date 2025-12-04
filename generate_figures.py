import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", font_scale=1.2)

# ---------------------------------------------------
# Simulated data for 30 responses (GPT, Gemini, Claude)
# ---------------------------------------------------

np.random.seed(42)

models = ["GPT", "Gemini", "Claude"]
num_samples = 30

# CSR similarity (cosine) — based on your empirical findings
csr_means = {"GPT": 0.86, "Gemini": 0.56, "Claude": 0.92}
csr_stds  = {"GPT": 0.05, "Gemini": 0.08, "Claude": 0.03}

csr_data = []

for m in models:
    samples = np.clip(
        np.random.normal(csr_means[m], csr_stds[m], num_samples),
        0.0, 1.0
    )
    csr_data.extend([(m, v) for v in samples])

csr_df = pd.DataFrame(csr_data, columns=["Model", "CSR"])

# ---------------------------------------------------
# CEI (Cognitive Entropy Index) — CT vs Non-CT
# ---------------------------------------------------

cei_no_ct_mean = 0.42
cei_ct_mean    = 0.34     # 約18%低下
cei_std        = 0.04

cei_df = pd.DataFrame({
    "Condition": ["No CT"] * num_samples + ["CT"] * num_samples,
    "CEI": np.concatenate([
        np.random.normal(cei_no_ct_mean, cei_std, num_samples),
        np.random.normal(cei_ct_mean, cei_std, num_samples)
    ])
})

# ---------------------------------------------------
# Five metrics comparison across models
# R, CEI, SVF, RS, PCS
# ---------------------------------------------------

metrics = ["R", "CEI", "SVF", "RS", "PCS"]

metric_values = {
    "GPT":    [0.86, 0.21, 0.33, 0.74, 0.70],
    "Gemini": [0.56, 0.31, 0.52, 0.48, 0.45],
    "Claude": [0.92, 0.19, 0.28, 0.82, 0.79]
}

metric_rows = []

for model in models:
    for metric, value in zip(metrics, metric_values[model]):
        metric_rows.append([model, metric, value])

metric_df = pd.DataFrame(metric_rows, columns=["Model", "Metric", "Value"])

# ---------------------------------------------------
# FIGURE 1: CSR distribution
# ---------------------------------------------------

plt.figure(figsize=(8,5))
sns.violinplot(x="Model", y="CSR", data=csr_df, palette="Set2")
sns.swarmplot(x="Model", y="CSR", data=csr_df, color="black", alpha=0.5)
plt.title("CSR Distribution Across Models")
plt.tight_layout()
plt.savefig("csr_distribution.png", dpi=300)
plt.savefig("csr_distribution.pdf")
plt.close()

# ---------------------------------------------------
# FIGURE 2: CEI reduction (CT vs Non-CT)
# ---------------------------------------------------

plt.figure(figsize=(8,5))
sns.boxplot(x="Condition", y="CEI", data=cei_df, palette="Set3")
sns.swarmplot(x="Condition", y="CEI", data=cei_df, color="black", alpha=0.5)
plt.title("CEI Reduction With Contractual Tags (CT)")
plt.tight_layout()
plt.savefig("cei_reduction.png", dpi=300)
plt.savefig("cei_reduction.pdf")
plt.close()

# ---------------------------------------------------
# FIGURE 3: Five metrics comparison
# ---------------------------------------------------

plt.figure(figsize=(9,6))
sns.barplot(x="Metric", y="Value", hue="Model", data=metric_df, palette="Set1")
plt.title("Cross-Model Comparison of Five Structural Metrics")
plt.legend(title="Model")
plt.tight_layout()
plt.savefig("metrics_comparison.png", dpi=300)
plt.savefig("metrics_comparison.pdf")
plt.close()

print("All figures generated: csr_distribution.*, cei_reduction.*, metrics_comparison.*")


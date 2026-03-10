import joblib
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# Load datasets
train = pd.DataFrame(joblib.load("monitoring/x_train.pkl"))
prod = pd.DataFrame(joblib.load("monitoring/x_test.pkl"))

# Ensure columns match
prod.columns = train.columns

drift_scores = []

# Compute KS test
for col in train.columns:
    stat, pvalue = ks_2samp(train[col], prod[col])
    drift_scores.append(pvalue)

# Plot results
plt.figure(figsize=(10,5))
plt.bar(train.columns, drift_scores)

plt.xticks(rotation=90)
plt.axhline(y=0.05, linestyle="--")
plt.title("Feature Drift Detection (KS Test)")
plt.ylabel("p-value")

plt.tight_layout()
plt.savefig("monitoring/drift_plot.png")
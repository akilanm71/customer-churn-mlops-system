import joblib
import pandas as pd
from scipy.stats import ks_2samp

train = pd.DataFrame(joblib.load("x_train.pkl"))
prod = pd.DataFrame(joblib.load("x_test.pkl"))

for col in train.columns:
    stat, pvalue = ks_2samp(train[col], prod[col])

    if pvalue < 0.05:
        print(f"Drift detected in {col}")




from evidently import Report
from evidently.presets import DataDriftPreset
import joblib
import pandas as pd

train = pd.DataFrame(joblib.load("x_train.pkl"))
prod = pd.DataFrame(joblib.load("x_test.pkl"))

report = Report(metrics=[DataDriftPreset()])

report.run(
    reference_data=train,
    current_data=prod
)

report.save_html("drift_report.html")

print("Drift report generated")
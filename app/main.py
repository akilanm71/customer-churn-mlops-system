from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os


app = FastAPI(title="Churn_prediction")
## Model Setup loading Data
@app.get("/")
def home():
    return {"message": "API running"}
## Churn Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "churn_lightgbm.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "features.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

class CustomerInput(BaseModel):
    
    Gender: int
    Age: float
    Married: int
    Number_of_Dependents: float
    Referred_a_Friend: int
    Number_of_Referrals: float
    Tenure_in_Months: float
    Phone_Service: int
    Avg_Monthly_Long_Distance_Charges: float
    Multiple_Lines: int
    Internet_Service: int
    Avg_Monthly_GB_Download: float
    Online_Security: int
    Online_Backup: int
    Device_Protection_Plan: int
    Premium_Tech_Support: int
    Streaming_TV: int
    Streaming_Movies: int
    Streaming_Music: int
    Unlimited_Data: int
    Contract: int
    Paperless_Billing: int
    Monthly_Charge: float
    Satisfaction_Score: float

    Offer: str
    Internet_Type: str
    City_Grouped: str
    Payment_Method: str 


## Churn Pred Preprocessing
def preprocess_input(input_dict, feature_columns):
    
    df = pd.DataFrame([input_dict])

    scaler_cols = scaler.feature_names_in_

    df[scaler_cols] = scaler.transform(df[scaler_cols])
    
    cat_cols = ["Offer", "Internet Type", "City_Grouped", "Payment Method"]
    
    df = pd.get_dummies(df, columns=cat_cols)
    
    df = df.reindex(columns=feature_columns, fill_value=0)
    
    bool_cols = df.select_dtypes(include=["bool", "uint8"]).columns
    df[bool_cols] = df[bool_cols].astype(int) 
    
    return df


@app.post("/predict")
def predict(customer: CustomerInput):
    
    input_dict = customer.model_dump()  
    processed_df = preprocess_input(input_dict,feature_columns)
    
    prediction = model.predict(processed_df)[0]
    probability = model.predict_proba(processed_df)[0][1]
    
    
    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(probability)
    }
import os
import pandas as pd
import numpy as np

from preprocessing.feature_engineering import extract_features
from preprocessing.encoding import apply_encoders
from preprocessing.scaling import apply_scaler

from inference_pipeline.model_loader import load_artifacts
from inference_pipeline.anomaly_scoring import compute_anomaly_score
from inference_pipeline.fraud_inference import predict_fraud
from risk_engine.rule_engine import RuleEngine
from risk_engine.decision_layer import DecisionLayer

class FraudSystem:
    def __init__(self):
        # Load all artifacts
        self.model, self.scaler, encoders_dict, self.anomaly_model = load_artifacts()
        self.oo = encoders_dict["oo"]
        self.enc = encoders_dict["enc"]

        self.rules = RuleEngine()
        self.decision = DecisionLayer()

    def process_transaction(self, transaction_dict):
        """
        Processes a single transaction dict and returns the decision.
        Matches the exact logic from the notebook `FraudSystem.predict()`.
        """
        # Convert single dict to DataFrame for preprocessing
        raw_df = pd.DataFrame([transaction_dict])

        # Enforce exact column order from the original CSV to avoid scaler warnings on inference
        expected_cols = ['trans_date_trans_time', 'cc_num', 'merchant', 'category', 'amt', 'first', 'last', 'gender', 'street', 'city', 'state', 'zip', 'lat', 'long', 'city_pop', 'job', 'dob', 'trans_num', 'unix_time', 'merch_lat', 'merch_long']
        for c in expected_cols:
            if c not in raw_df.columns:
                raw_df[c] = 0  # Fill missing dummy columns dropped later anyway

        raw_df = raw_df[expected_cols]

        cols_to_drop = ['Unnamed: 0', 'cc_num', 'first', 'last', 'street', 'trans_num', 'zip']
        raw_df = raw_df.drop(columns=cols_to_drop, errors='ignore')

        # Feature Engineering (Extract features)
        df_features = extract_features(raw_df)

        # Apply Encoders
        df_encoded = apply_encoders(df_features, self.oo, self.enc)

        # Align columns properly - assuming no "is_fraud" in live inference
        if "is_fraud" in df_encoded.columns:
            df_encoded = df_encoded.drop(columns=["is_fraud"])

        # Scale features
        x_scaled = apply_scaler(df_encoded, self.scaler)

        # Anomaly Score
        anomaly_score = compute_anomaly_score(self.anomaly_model, x_scaled)[0]

        # Append anomaly score to features (as done before SMOTE in train.py)
        x_scaled_with_anomaly = np.hstack([x_scaled, [[anomaly_score]]])

        # Fraud Probability (NN)
        fraud_prob = predict_fraud(self.model, x_scaled_with_anomaly[0])

        # Rule Engine expects raw inputs that INCLUDE 'hour' and 'distance'.
        # Since extract_features mutates df_features to include them, we should pass the first row.
        rule_score, flags = self.rules.evaluate_extended(df_features.iloc[0].to_dict(), fraud_prob, anomaly_score)

        # Decision Layer
        decision_status = self.decision.decide(fraud_prob, anomaly_score, rule_score)

        return {
            "fraud_probability": float(fraud_prob),
            "anomaly_score": float(anomaly_score),
            "rule_score": int(rule_score),
            "flags": flags,
            "decision": decision_status
        }

if __name__ == "__main__":
    import math

    # Sample minimal execution test
    print("Loading the Fraud System...")
    try:
        system = FraudSystem()
        
        # Fake transaction matching roughly the expected schema
        sample_txn = {
            "trans_date_trans_time": "2023-11-20 03:00:00",
            "amt": 6000.50,
            "gender": "F",
            "category": "shopping_net",
            "state": "NY",
            "merchant": "fraud_ltd",
            "city": "New York",
            "job": "Data Scientist",
            "dob": "1990-01-01",
            "lat": 40.7128,
            "long": -74.0060,
            "merch_lat": 42.7128,
            "merch_long": -72.0060,
            "city_pop": 150000,
            "unix_time": 1690000000
        }
        
        print("\nProcessing sample transaction:", sample_txn)
        result = system.process_transaction(sample_txn)
        
        print("\n--- System Decision ---")
        for k, v in result.items():
            print(f"{k}: {v}")

    except Exception as e:
        print(f"Error initializing system or running inference: {e}")
        print("Note: If artifacts do not exist, run fraud_detection_system/training_pipeline/train.py first.")

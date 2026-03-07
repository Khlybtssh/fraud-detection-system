import os
import joblib
import torch
from models.fraud_nn import FraudNN
from config import config, ARTIFACTS_DIR

def load_artifacts():
    """ Load trained models, scalers, and encoders from artifacts folder """
    # Load PyTorch model
    model = FraudNN(**config["model"])
    model.load_state_dict(torch.load(os.path.join(ARTIFACTS_DIR, "model.pt"), map_location=torch.device('cpu')))
    model.eval()

    # Load scikit-learn artifacts
    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
    encoders = joblib.load(os.path.join(ARTIFACTS_DIR, "encoders.pkl"))
    anomaly_model = joblib.load(os.path.join(ARTIFACTS_DIR, "anomaly_model.pkl"))

    return model, scaler, encoders, anomaly_model

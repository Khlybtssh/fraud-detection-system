import os
import joblib
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config, ARTIFACTS_DIR
from training_pipeline.data_loader import load_data
from preprocessing.feature_engineering import extract_features
from preprocessing.encoding import fit_encoders, apply_encoders
from preprocessing.scaling import fit_scaler, apply_scaler
from training_pipeline.anomaly_training import train_anomaly_model, apply_anomaly_scores
from models.fraud_nn import FraudNN
from training_pipeline.dataset import FraudDataset
from training_pipeline.trainer import Trainer
from utils.metrics import evaluate_model

def run_training():
    print("Loading data...")
    train_df, test_df = load_data(config["data"]["train_path"], config["data"]["test_path"])

    print("Extracting features...")
    train_df = extract_features(train_df)
    test_df = extract_features(test_df)

    print("Fitting and applying encoders...")
    oo, enc = fit_encoders(train_df)
    train_df = apply_encoders(train_df, oo, enc)
    test_df = apply_encoders(test_df, oo, enc)

    X = train_df.drop("is_fraud", axis=1)
    y = train_df["is_fraud"]

    print("Fitting and applying scaler...")
    ss = fit_scaler(X)
    X = apply_scaler(X, ss)
    X_test = apply_scaler(test_df.drop("is_fraud", axis=1), ss)
    y_test = test_df["is_fraud"]

    print("Splitting validation set...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training Anomaly Model...")
    IF = train_anomaly_model(X_train, config["anomaly"])
    
    # Get anomaly scores
    train_scores = apply_anomaly_scores(IF, X_train)
    val_scores = apply_anomaly_scores(IF, X_val)
    test_scores = apply_anomaly_scores(IF, X_test)

    # Append anomaly scores as a feature
    X_train = np.hstack([X_train, train_scores.reshape(-1, 1)])
    X_val = np.hstack([X_val, val_scores.reshape(-1, 1)])
    X_test = np.hstack([X_test, test_scores.reshape(-1, 1)])

    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print("Preparing datasets for PyTorch...")
    train_dataset = FraudDataset(X_train_smote, y_train_smote)
    val_dataset = FraudDataset(X_val, y_val)
    test_dataset = FraudDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config["data"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["data"]["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["data"]["batch_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = FraudNN(**config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer"]["lr"])
    loss_fn = nn.BCEWithLogitsLoss()

    trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, device=device)

    print("Training Neural Network...")
    for epoch in range(config["training"]["epochs"]):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, precision, recall, f1, auc, probs, preds, targets = evaluate_model(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | AUC: {auc:.4f}")

    print("Saving artifacts...")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(ARTIFACTS_DIR, "model.pt"))
    joblib.dump(ss, os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
    joblib.dump({"oo": oo, "enc": enc}, os.path.join(ARTIFACTS_DIR, "encoders.pkl"))
    joblib.dump(IF, os.path.join(ARTIFACTS_DIR, "anomaly_model.pkl"))

    print("Training complete and artifacts saved.")

if __name__ == "__main__":
    run_training()

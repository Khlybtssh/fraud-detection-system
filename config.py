import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

config = {
    "data": {
        "batch_size": 512,
        "train_path": r"C:\Users\Osama Youssef\Downloads\AI Projects\fraud-detection-system\data\fraudTrain.csv",
        "test_path": r"C:\Users\Osama Youssef\Downloads\AI Projects\fraud-detection-system\data\fraudTest.csv"
    },
    "model": {
        "input_dim": 77,
        "hidden_dims": [128, 64],
        "dropout": 0.2
    },
    "optimizer": {
        "type": "adam",
        "lr": 0.001
    },
    "training": {
        "epochs": 20
    },
    "anomaly": {
        "n_estimators": 200,
        "max_samples": 10000,
        "contamination": 0.006,
        "random_state": 42
    }
}

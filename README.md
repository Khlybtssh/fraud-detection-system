# Fraud Detection System

A production-ready anomaly detection and fraud classification system for financial transactions. 

This project was securely refactored from a Jupyter Notebook pipeline into a robust Python architecture with separated systems for data processing, model training, artifact management, and live inference.

## Architecture Overview

The system utilizes two primary models running in tandem to evaluate transaction risk:
1. **Anomaly Detection Layer**: An `IsolationForest` model identifying statistically abnormal behaviors or outlier transactions lacking historical precedent.
2. **Fraud Neural Network**: A PyTorch deep learning classifier balanced via `SMOTE` generating a statistical probability of transactional fraud based on encoded characteristics.
3. **Risk Engine**: A hardcoded deterministic rule engine checking transactions against domain-specific triggers (e.g., Night Owl risks, Geographic risks). Let's the system override fuzzy ML boundaries in exceptional circumstances.

These outputs feed into a unified **Decision Layer** which determines if a transaction is `APPROVE`, `REVIEW`, or `DECLINE`.

### Directory Layout

```
fraud_detection_system/
│
├── config.py                     # Centralized hyperparameters & paths
├── main.py                       # Direct Python entry point for single-transaction inference
├── app.py                        # FastAPI REST API server for HTTP-based inference
│   
├── artifacts/                    # Saved output directory for trained ML weights
│   ├── anomaly_model.pkl         # Fitted IsolationForest instance
│   ├── encoders.pkl              # Fitted OneHot/Ordinal encoders dict
│   ├── model.pt                  # PyTorch model weights dict
│   └── scaler.pkl                # Fitted StandardScaler instance
│
├── training_pipeline/            # Executable scripts for dataset ingestion & training
│   ├── train.py                  # Entry script combining pipeline components
│   ├── anomaly_training.py       # Wrapper for Isolation Forest fits
│   ├── dataset.py                # PyTorch custom Dataset class
│   ├── data_loader.py            # pandas loading and explicit column drops
│   └── trainer.py                # PyTorch optimization loops and validation logic
│
├── inference_pipeline/           # Executable components for evaluating a parsed transaction
│   ├── anomaly_scoring.py        # Applies fitted Isolation Forest
│   ├── fraud_inference.py        # Evaluates PyTorch classifier probabilties
│   └── model_loader.py           # Loads .pkl and .pt weights safely from artifacts
│
├── preprocessing/                # Shared stateless logic ensuring exact parity 
│   ├── encoding.py               # Categorical mapping definitions
│   ├── feature_engineering.py    # Time, Distance, Haversine, and coordinate parsing
│   └── scaling.py                # StandardScaler interfaces
│
├── models/                       # Network architectures 
│   └── fraud_nn.py               # PyTorch `FraudNN` sequential module
│
├── risk_engine/                  # Deterministic evaluators 
│   ├── rule_engine.py            # Static condition flags (e.g. `HIGH_AMOUNT`)
│   └── decision_layer.py         # Evaluates all thresholds to output APPROVE/REVIEW
│
└── utils/
    └── metrics.py                # Standard ML calculation functions (F1, AUC, Recall)
```

## Setup & Execution

### 1. Training the System

Before you can run predictions, you must execute the training pipeline to generate the necessary artifacts. Ensure your CSV data paths in `config.py` accurately reflect your local dataset location.

```bash
cd fraud_detection_system
python training_pipeline/train.py
```

**Training Process (`train.py`):**
1. Loads the defined datasets.
2. Parses complex features like geographical `haversine` distance and datetime offsets via `feature_engineering.py`.
3. Handles `OneHotEncoding` and `OrdinalEncoding` for categorical identifiers.
4. Scales remaining features using `StandardScaler`.
5. Fits the unsupervised `IsolationForest` anomaly model.
6. Addresses class imbalances using `SMOTE`.
7. Instantiates and optimizes the target `FraudNN` PyTorch classifier on CUDA (if available).
8. Dumps all successfully fitted components into `artifacts/`.

### 2. Live Inference

With the artifacts compiled, you have two options for running live inference:

#### Option A: REST API (`app.py`) — Recommended

The system exposes a **FastAPI** REST API for HTTP-based transaction evaluation. Start the server with:

```bash
cd fraud_detection_system
uvicorn app:app --reload
```

The API will be available at `http://127.0.0.1:8000`. Interactive docs are auto-generated at `/docs` (Swagger UI) and `/redoc`.

#### API Endpoints

| Method   | Endpoint                          | Description                                  |
|----------|-----------------------------------|----------------------------------------------|
| `GET`    | `/`                               | Health check — returns a welcome message     |
| `GET`    | `/transactions`                   | List all recorded transactions               |
| `GET`    | `/transactions/{transaction_id}`  | Retrieve a specific transaction by ID        |
| `POST`   | `/transactions`                   | Submit a new transaction for evaluation      |
| `DELETE` | `/transactions/{transaction_id}`  | Delete a transaction by ID                   |

#### Example Requests

**List all transactions:**
```bash
curl http://127.0.0.1:8000/transactions
```

**Get a specific transaction:**
```bash
curl http://127.0.0.1:8000/transactions/1
```

**Submit a new transaction:**
```bash
curl -X POST "http://127.0.0.1:8000/transactions?amount=500.00&merchant=Amazon"
```
Transactions with `amount < 1000` are automatically `approved`; amounts `>= 1000` are `flagged` for review.

**Delete a transaction:**
```bash
curl -X DELETE http://127.0.0.1:8000/transactions/2
```

#### Option B: Direct Python (`main.py`)

`main.py` serves as a standalone entry point for evaluating transactions directly via Python without spinning up a server.

```bash
cd fraud_detection_system
python main.py
```

**Inference Process (`main.py`):**
1. System initializes and caches weights via `model_loader.py`.
2. A raw dictionary transaction is ingested (`process_transaction`).
3. The feature columns are aligned to exactly match what the training loop observed.
4. Unnecessary reference columns (`cc_num`, `first`, `last`, etc) are dropped.
5. Missing encodings are filled, features are extracted, and arrays are scaled matching the logic inside `preprocessing`.
6. PyTorch forward passes generate the fraud probability while `IsolationForest` returns the severity of the transactional anomaly.
7. Both metadata sources route into the Rule Engine logic.
8. The final decision string is returned via the Decision Layer algorithm.

## Configuration

You can easily adjust hyperparameters or model architecture via `config.py` without mutating the operational code:
```python
"model": {
    "input_dim": 77,
    "hidden_dims": [128, 64],
    "dropout": 0.2
},
"optimizer": {
    "type": "adam",
    "lr": 0.001
}
```

## Dependencies

The API layer requires **FastAPI** and **Uvicorn**. Install them alongside the existing requirements:

```bash
pip install fastapi uvicorn
pip install -r requirements.txt
```

## Adding New Features/Rules
- To add a new ML feature transformation, update `preprocessing/feature_engineering.py`.
- To establish a new company policy or business logic constraint, implement the flag in `risk_engine/rule_engine.py`.
- To add a new API endpoint, define the route in `app.py`.

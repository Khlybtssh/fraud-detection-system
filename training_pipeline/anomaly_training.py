from sklearn.ensemble import IsolationForest

def train_anomaly_model(X_train, config):
    """
    Trains the IsolationForest anomaly detection model.
    """
    IF = IsolationForest(
        n_estimators=config["n_estimators"],
        max_samples=config["max_samples"],
        contamination=config["contamination"],
        n_jobs=-1,
        random_state=config["random_state"]
    )
    IF.fit(X_train)
    return IF

def apply_anomaly_scores(model, X):
    """
    Returns decision function scores.
    """
    return model.decision_function(X)

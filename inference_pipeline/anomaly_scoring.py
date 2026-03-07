def compute_anomaly_score(anomaly_model, x_features):
    """ Compute anomaly score for a single or multiple transactions """
    return anomaly_model.decision_function(x_features)

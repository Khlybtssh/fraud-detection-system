class DecisionLayer:
    def __init__(self, fraud_threshold=0.9, anomaly_threshold=-0.2, rule_threshold=3):
        self.fraud_threshold = fraud_threshold
        self.anomaly_threshold = anomaly_threshold
        self.rule_threshold = rule_threshold

    def decide(self, fraud_prob, anomaly_score, rule_score):
        if fraud_prob > self.fraud_threshold:
            return "DECLINE"
        elif anomaly_score < self.anomaly_threshold:
            return "REVIEW"
        elif rule_score > self.rule_threshold:
            return "REVIEW"
        else:
            return "APPROVE"

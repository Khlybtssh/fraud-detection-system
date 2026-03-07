def rule_engine(transaction, fraud_prob, anomaly_score):
    risk_points = 0
    flags = []

    # القاعدة 1: المبالغ الكبيرة جداً (High Amount Rule)
    if transaction['amt'] > 5000:
        risk_points += 3
        flags.append("HIGH_AMOUNT")

    # القاعدة 2: معاملات الفجر (Night Owl Rule)
    if 2 <= transaction['hour'] <= 5 and transaction['amt'] > 500:
        risk_points += 2
        flags.append("NIGHT_TRANSACTION")

    # القاعدة 3: المسافات البعيدة جداً (Geographic Risk)
    if transaction['distance'] > 200:
        risk_points += 2
        flags.append("LARGE_DISTANCE")
        
    # دمج مخرجات الـ Anomaly Detection (Isolation Forest)
    if anomaly_score < -0.5: # القيم السالبة جداً تعني شذوذ قوي
        risk_points += 1
        flags.append("ANOMALOUS_BEHAVIOR")

    return risk_points, flags

class RuleEngine:
    # A simple wrapper for the above rule_engine to maintain API compatibility if needed
    def evaluate(self, transaction, fraud_prob=0, anomaly_score=0):
        # We pass default 0 since sometimes evaluate takes only the raw transaction
        # Wait, in the notebook FraudSystem.predict calls:
        # self.rules.evaluate(raw_transaction)
        # which means it evaluated something else OR it expected anomaly_score. I will map it strictly.
        # Actually in notebook `self.rules = RuleEngine()` isn't fully defined but `rule_engine` is defined right before it.
        # Let's adjust it to take all three.
        pass

    def evaluate_extended(self, transaction, fraud_prob, anomaly_score):
        return rule_engine(transaction, fraud_prob, anomaly_score)

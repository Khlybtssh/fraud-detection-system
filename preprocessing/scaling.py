from sklearn.preprocessing import StandardScaler

def fit_scaler(X):
    """ Fits a standard scaler to the features """
    ss = StandardScaler()
    ss.fit(X)
    return ss

def apply_scaler(X, ss):
    """ Applies the fitted scaler """
    return ss.transform(X)

from sklearn.preprocessing import MinMaxScaler

def preprocess(X_train, X_test):
    scaler = MinMaxScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)

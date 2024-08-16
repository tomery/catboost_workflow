# src/preprocessor.py
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def fit_transform(self, X_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        return X_train_scaled
    
    def transform(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        return X_test_scaled

# src/model.py
from catboost import CatBoostClassifier

class Model:
    def __init__(self, config, params=None):
        if params is None:
            params = {
                'iterations': config['model']['iterations'],
                'learning_rate': config['model']['learning_rate'],
                'eval_metric': config['model']['eval_metric'],
                'verbose': False
            }
        self.model = CatBoostClassifier(**params)
    
    def train(self, X_train, y_train, X_valid, y_valid):
        self.model.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=False)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def save_model(self, path):
        self.model.save_model(path)
    
    def get_model(self):
        return self.model

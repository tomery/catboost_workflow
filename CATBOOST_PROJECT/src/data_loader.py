# src/data_loader.py
import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, config):
        self.data_path = config['data']['path']
        self.test_size = config['data']['test_size']
        self.random_state = config['data']['random_state']
        self.target_column = config['data']['target_column']
    
    def load_data(self):
        df = pd.read_csv(self.data_path)
        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test

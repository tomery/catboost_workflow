# main.py
import json
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.model import Model
from src.optimizer import Optimizer
from src.visualizer import Visualizer
import os

def main():
    # Load configuration
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    
    # Load data
    data_loader = DataLoader(config)
    X_train, X_test, y_train, y_test = data_loader.load_data()
    
    # Preprocess data
    preprocessor = Preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Split training data into training and validation sets
    from sklearn.model_selection import train_test_split
    X_train_final, X_valid, y_train_final, y_valid = train_test_split(
        X_train_processed, y_train, test_size=config['data']['test_size'], random_state=config['data']['random_state'], stratify=y_train
    )
    
    # Hyperparameter Optimization
    optimizer = Optimizer(X_train_final, y_train_final, X_valid, y_valid, config)
    best_params = optimizer.optimize()
    
    # Update model parameters with best hyperparameters
    model_params = best_params
    model_params['eval_metric'] = config['model']['eval_metric']
    model_params['verbose'] = False
    model_params['random_state'] = config['data']['random_state']
    
    # Train final model
    model = Model(config, params=model_params)
    model.train(X_train_final, y_train_final, X_valid, y_valid)
    
    # Save model
    os.makedirs(os.path.dirname(config['output']['model_path']), exist_ok=True)
    model.save_model(config['output']['model_path'])
    print(f"Model saved at {config['output']['model_path']}")
    
    # Visualizations and Reports
    visualizer = Visualizer(model.get_model(), X_test_processed, y_test, config['output']['reports_path'])
    visualizer.generate_all_reports()

if __name__ == "__main__":
    main()

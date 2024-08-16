# generate_config.py
import json
import os

config = {
    "data": {
        "path": "data/data.csv",
        "test_size": 0.2,
        "random_state": 42,
        "target_column": "target"
    },
    "model": {
        "iterations": 1000,
        "learning_rate": 0.1,
        "eval_metric": "AUC"
    },
    "optuna": {
        "n_trials": 50,
        "timeout": 3600
    },
    "output": {
        "model_path": "output/catboost_model.cbm",
        "reports_path": "output/reports/"
    }
}

os.makedirs('config', exist_ok=True)

with open('config/config.json', 'w') as f:
    json.dump(config, f, indent=4)

print("Configuration file generated at 'config/config.json'")

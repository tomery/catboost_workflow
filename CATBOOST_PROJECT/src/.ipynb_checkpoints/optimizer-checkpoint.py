# src/optimizer.py
import optuna
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

class Optimizer:
    def __init__(self, X_train, y_train, X_valid, y_valid, config):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.n_trials = config['optuna']['n_trials']
        self.timeout = config['optuna']['timeout']
        self.random_state = config['data']['random_state']
    
    def objective(self, trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_strength': trial.suggest_float('random_strength', 1e-9, 10, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
            'task_type': 'CPU',
            'eval_metric': 'AUC',
            'verbose': False,
            'random_state': self.random_state
        }
        
        model = CatBoostClassifier(**params)
        model.fit(self.X_train, self.y_train, eval_set=[(self.X_valid, self.y_valid)], early_stopping_rounds=50, verbose=False)
        preds = model.predict_proba(self.X_valid)[:, 1]
        auc = roc_auc_score(self.y_valid, preds)
        return auc
    
    def optimize(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trials, timeout=self.timeout)
        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        
        return trial.params

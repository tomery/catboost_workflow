**This one includes a classification modeling flow using Catboost model and handels several steps:**
* Preprocessing the data
* Modeling with hyperparameter tuning (using Optuna )
* Post analysis and Exaplanability review

**One should follow the steps: **
0. use pip install -r requirements.txt - to have the packages needed.
1. Run the following generate_config.py in order to generate a config.json file in 'config' folder
2. Run the following generate_data.py in order to create the data set - to be located in 'data' folder
3. The main script to run modeling, optimization and visualization: main.py

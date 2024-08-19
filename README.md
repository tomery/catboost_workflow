## Simple modeling workflow of Classification problem using Catboost along with data preprocessing,modeling with hyperparameter tuning and post analysis of model
-------------------------------
**This repo includes a classification modeling flow 'Tutorial' using Catboost model,that covers the following:**
* Preprocessing the data - just for completeness even though Catboost doesn't require feature 'normalization'/'standardization'
* Modeling with hyperparameter tuning (using Optuna )
* Post analysis and Exaplanability review

**One should follow the steps:**
0. use pip install -r requirements.txt - to have the packages needed.
1. Run the following generate_config.py in order to generate a config.json file in 'config' folder - UPDATE IT AS NEEDED!
2. Run the following generate_data.py in order to create the data set - to be located in 'data' folder
3. The main script to run modeling, optimization and visualization: main.py

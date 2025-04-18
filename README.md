## Overview

The project performs a binary classification task to predict loan status (0 for non-default, 1 for default) using the CatBoost algorithm. You can get the dataset from the Playground Series Season 4 Episode 10 kaggle competition to train and evaluate a predictive model, and even find my notebook there. It includes data loading, preprocessing, feature engineering, model training using stratified k-fold cross-validation, and output file. Additional stuff are EDA and feature importance analysis using SHAP values.
## Requirements
Libraries:
    -numpy
    -pandas
    -matplotlib
    -catboost
    -scikit-learn
    -seaborn
    -shap

## Input Data

The script expects the following CSV files to be present in the specified paths:
1.  `/kaggle/input/playground-series-s4e10/train.csv`: The training data from the competition.
2.  `/kaggle/input/playground-series-s4e10/test.csv`: The test data
3.  `/kaggle/input/credit-risk-dataset-csv/credit_risk_dataset.csv`: An additional dataset used for augmenting the training data

1.  **Import Libraries:** Imports all necessary python libraries
2.  **Load Data:** Loads the training, test, and original credit risk datasets into pandas DataFrames
3.  **Initial Data Exploration**
4.  **Missing Value Handling** 
5.  **Data Augmentation:** Concatenates the original dataset to the training dataset to increase the training data size. An `id` column is created for the combined training data.
6.  **Combined Data Preparation:** Concatenates the augmented training data and the test data into a single DataFrame `df` for consistent preprocessing.
7.  **Feature Engineering:**
    - Creating new features:
      loantoincome`: Ratio of `loan_amnt` to `person_income` (converted to categorical).
      loan_percent_incometoincome`: Ratio of `loan_percent_income` to `person_income` (converted to categorical after ensuring numeric types).
      person_age_to_person_income`: Ratio of `person_age` to `person_income` (converted to categorical).
      person_emp_length_to_person_age`: Ratio of `person_emp_length` to `person_age` (converted to categorical).
      loan_int_rate_to_loan_amnt`: Ratio of `loan_int_rate` to `loan_amnt` (converted to categorical).
    - Converts specified columns (`person_age`, `person_income`, `person_emp_length`, `cb_person_cred_hist_length`, `loan_int_rate`, `loan_amnt`, `loan_percent_income`) to numeric types to ensure proper calculations.
8.  **Categorical Encoding:**
    - Replaced categorical string values in `person_home_ownership`, `loan_intent`, `loan_grade`, and `cb_person_default_on_file` with numerical codes and converted them to the `category` data type.
    - `loan_int_rate` and `loan_percent_income` are converted to categorical types
    - Creating feature `person_home_ownership_income` by combining `person_home_ownership` and `person_income` and then factorizing the result into categorical codes.
9.  **Data Splitting:** Splitting the combined `df` back into the `train` and `test` DataFrames based on the original num of rows in the training data. The `loan_status` column is dropped from the `test` set 
10. **Model Training and Evaluation:**
    - created list of categorical features.
    - created lists to store AUC scores and predictions from each fold of cross-validation.
    - Stratified K-Fold cross-validation with 5 splits, shuffling the data and using a random state for reproducibility.
    - Iterating through each fold:
        - Splitting the training data into training and validation sets based on the fold indices.
        - Extracting the features (`X_train`, `X_valid`, `X_test`) and the target variable (`y_train`, `y_valid`).
        - Creating CatBoost `Pool` objects for efficient handling of categorical feature
        - Initializing a `CatBoostClassifier` with hyperparameters:
            - `loss_function='Logloss'` for binary classification
            - `eval_metric='AUC'` to evaluate model performance
            - `learning_rate=0.1` to control the step size during gradient descent.
            - `iterations=1000` as the maximum number of boosting iterations
            - `depth=7` as the depth of the trees
            - `random_strength=1` to prevent overfitting
            - `l2_leaf_reg=20` for L2 regularization
            - `task_type='CPU'`
            - `random_seed=42`
            - `verbose=False` to suppress training output
            - `leaf_estimation_method="Gradient"`, `bootstrap_type="Bernoulli"`, `grow_policy="SymmetricTree"`
            - `early_stopping_rounds=50` to stop training if the validation AUC doesn't improve.
            - `use_best_model=True` to use the model from the best iteration.
            - `score_function='Cosine'` as an additional scoring function.
            - `subsample=0.8` for stochastic gradient boosting.
            - `od_pval=0.1` for another early stopping criterion.
        - Fitting CatBoost model on training data and evaluating it on validation data
        - Predicting probabilities for the positive classon the training, validation, and test sets.
        - Calculating the AUC score for the validation set and the training set.
        - Appends the test set predictions and the validation AUC to their respective lists.
        - Printing AUC score for each fold
    - Calculating overall mean AUC and its standard deviation across all folds.
11. **EDA:**
    - creating a correlation heatmap of the training data using `seaborn` to visualize the relationships between features.
    - Creating a pair plot of the training data to visualize pairwise relationships and distributions of features.
    - creating a box plot and a catplot to visualize the distribution of `loan_amnt` for each `loan_status`.
12. **Feature Importance Analysis:**
    - Initializing a SHAP TreeExplainer using the trained CatBoost model from the last fold.
    - Calculating SHAP values for the training data.
    - Generating a summary bar plot of the mean absolute SHAP values to show the relative importance of each feature in the model's predictions.
13. **Submission File Generation**


## Notes

GridSearchCV/RandomizedSearchCV could be used for improvement??.
Feature engineering steps aim to create potentially more informative features from the existing ones. The effectiveness of these engineered features can be evaluated through their impact on the model's performance and feature importance analysis.
Further feature engineering or better model selection?
SHAP values help in understanding the contribution of each feature to the model's output for individual predictions and provide a global measure of feature importance.
The `early_stopping_rounds` parameter helps to prevent overfitting
The `use_best_model=True` parameter ensures that the model from the iteration with the best validation score is used for prediction.


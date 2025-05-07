# Filename: lr_train_evaluate.py

# Author: Raha Musavi
# Matrikelnummer: 08022255354
# Thesis: Modeling the Presence and the Classification of Ezafe in Middle Persian Nominal Phrases 
# Date: 01 May 2025

"""
This script trains and evaluates the final Logistic Regression model
for predicting the presence of Ezafe. It loads the preprocessed and
feature-selected data, handles class imbalance, splits the data into
training and test sets, performs hyperparameter tuning using GridSearchCV,
trains the best model found, evaluates its performance using standard metrics,
and saves the trained model to a file.

Corresponds to Handling Class Imbalance (Section 5.5), Hyperparameter Tuning (Section 5.6),
Model Training (Section 5.7), and Evaluation (Section 5.9, 5.10) in Chapter 5 of the thesis.

Finally the results from this model are reflected in Table 11 of the thesis.
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler # Requires: pip install imblearn
import os
import joblib # Requires: pip install joblib # Used for saving/loading the trained model


# Suppress warnings (e.g., from specific solvers)
warnings.filterwarnings("ignore")

# --- Configuration ---
# Define input and output file paths.
# Assumes lr_feature_engineering.py has been run and created these files.
INPUT_FEATURES_CSV = "lr_final_features.csv"
INPUT_TARGET_CSV = "lr_target.csv"
OUTPUT_MODEL_PATH = "final_logistic_regression_model.pkl"

# Define model training parameters (Section 5.2, 5.7)
MAX_ITER = 500 # Maximum number of iterations for the solver to converge.

# Define data splitting parameters (Section 5.9, unchanged from Table 9, step 8)
TEST_SIZE = 0.3 # Proportion of the dataset to include in the test split.
RANDOM_STATE = 42 # Seed for the random number generator to ensure reproducibility of splits.

# Define Hyperparameter Grid for GridSearchCV (Section 5.6, unchanged from Table 9 Step 8)
# This grid specifies the range of hyperparameters to search over.
param_grid = {
    'C': [0.1, 1, 10, 100], # Regularization strength values to test. Lower C means stronger regularization.
    'solver': ['newton-cg', 'liblinear'] # Solvers suitable for this type of problem and dataset size.
}

# Define Cross-Validation strategy parameters (Section 5.6, 5.9)
CV_FOLDS = 10 # Number of folds for cross-validation during grid search (unchanged from Table 9 Step 8).
CV_SHUFFLE = True # Whether to shuffle the data before splitting into folds. Recommended with random_state.

# --- Main Processing Flow ---

if __name__ == "__main__":
    # Load the final feature set and target prepared by the feature engineering script.
    if not os.path.exists(INPUT_FEATURES_CSV) or not os.path.exists(INPUT_TARGET_CSV):
        print(f"Error: Input files '{INPUT_FEATURES_CSV}' or '{INPUT_TARGET_CSV}' not found.")
        print("Please run lr_feature_engineering.py first.")
        exit()

    X = pd.read_csv(INPUT_FEATURES_CSV)
    y = pd.read_csv(INPUT_TARGET_CSV).squeeze() #squeeze() to get a pandas Series for the target variable

    print(f"Loaded final features (X) shape: {X.shape}")
    print(f"Loaded target (y) shape: {y.shape}")


    # Handle Class Imbalance using RandomOverSampler (Section 5.5, unchanged from Table 9 Step 8)
    # Randomly samples the minority class with replacement to match the number
    # of samples in the majority class, balancing the dataset for training.
    oversampler = RandomOverSampler(random_state=RANDOM_STATE)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    print(f"\nResampled data shape: {X_resampled.shape}")
    print(f"Resampled target distribution:\n{y_resampled.value_counts()}")


    # Split the resampled data into training and test sets (Section 5.9, unchanged from Table 9 Step 8) (as it has proved to render better results than only on the train set).
    # A 70-30 split is used. Stratified splitting ensures that the proportion
    # of target classes is the same in both training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_resampled # Essential for maintaining balance in train/test splits
    )
    print(f"\nTrain data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")


    # Define the base Logistic Regression model (Section 5.2)
    # class_weight='balanced' is used to automatically adjust weights inversely
    # proportional to class frequencies in the input data (before resampling),
    # further helping to handle class imbalance during model fitting.
    model = LogisticRegression(max_iter=MAX_ITER, class_weight='balanced')


    # Define Cross-Validation strategy for GridSearchCV (Section 5.6, 5.9)
    # StratifiedKFold is used to ensure that each fold in the cross-validation
    # process has the same distribution of target classes as the full dataset.
    cv_strategy = StratifiedKFold(n_splits=CV_FOLDS, shuffle=CV_SHUFFLE, random_state=RANDOM_STATE)


    # Set up GridSearchCV for hyperparameter tuning (Section 5.6)
    # GridSearchCV exhaustively searches over the specified parameter grid
    # using the defined cross-validation strategy to find the best model parameters.
    # refit=True ensures the best model is retrained on the full training data after the search.
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv_strategy, # Use the defined CV strategy
        n_jobs=-1, # Uses all available CPU cores for parallel processing
        verbose=2, # Print progress updates during the search
        refit=True # Retrain the best model on the full training data after the search
    )

    # Train the model by fitting GridSearchCV on the training data.
    print("\nStarting GridSearchCV training for hyperparameter tuning...")
    grid_search.fit(X_train, y_train)

    # Print the best parameters and the best cross-validation score found (Section 5.6)
    print(f"\nBest parameters found by GridSearchCV: {grid_search.best_params_}")
    print(f"Best cross-validation mean score: {grid_search.best_score_}")

    # Evaluate the best model on the unseen test set (unchanged from Section 5.9, 5.10)
    # Get the best model found during the grid search.
    best_model = grid_search.best_estimator_
    # Make predictions on the test data.
    y_pred = best_model.predict(X_test)

    print("\n--- Test Set Evaluation ---")
    # Calculate and print overall accuracy.
    print("Accuracy:", accuracy_score(y_test, y_pred))
    # Print the classification report showing precision, recall, f1-score for each class (Table 11).
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # --- Save the trained model ---
    # Save the best trained model using joblib for future use or deployment.
    try:
        joblib.dump(best_model, OUTPUT_MODEL_PATH)
        print(f"\nFinal trained Logistic Regression model saved to '{OUTPUT_MODEL_PATH}'")
    except Exception as e:
        print(f"Error saving the trained model: {e}")

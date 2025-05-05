# Filename: lr_feature_engineering.py

"""
This script loads the cleaned nominal phrase data CSV, performs feature engineering
specific to the Logistic Regression model, including encoding categorical features,
standardizing numerical features, and calculating feature importance. It then selects
the most important features based on a defined threshold. The final feature set
and the target variable are saved to separate CSV files, which serve as input
for the model training script.

Corresponds to Feature Engineering (Section 5.4), Scaling (Section 5.4.5),
Feature Importance Analysis (Section 5.4.6), and Feature Selection steps in
Chapter 5 of the thesis.
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, OneHotEncoder
import os

# Suppress warnings (e.g., from solvers during initial fit)
warnings.filterwarnings("ignore")

# --- Configuration ---
# Define input and output file paths.
INPUT_CSV_PATH = "nominal_features_cleaned.csv"
OUTPUT_FEATURES_CSV = "lr_final_features.csv"
OUTPUT_TARGET_CSV = "lr_target.csv"
FEATURE_IMPORTANCE_CSV = "LR_feature_importance_all_features.csv"

# Define the threshold for feature selection based on absolute importance (Section 5.4.6, Table 9 Step 8)
# Features with absolute coefficient value below this threshold are removed.
IMPORTANCE_THRESHOLD = 0.15

# --- Main Processing Flow ---

if __name__ == "__main__":
    # Load the dataset generated in the preprocessing step
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: Input file not found: {INPUT_CSV_PATH}")
        print("Please run lr_preprocess_data.py first.")
        exit()

    nominals_df = pd.read_csv(INPUT_CSV_PATH)
    print(f"Loaded data from {INPUT_CSV_PATH}. Shape: {nominals_df.shape}")

    # --- Feature Engineering Steps (Based on Section 5.4 and Table 9 Step 8 logic) ---

    # Drop columns not used as features for LR (IDs, Lemmas, Nominal Head UPOS)
    # IDs and Lemmas are dropped as they don't contribute linguistic information (Section 5.4.1).
    # Nominal Head UPOS was removed as it did not significantly contribute (Section 5.4.1).
    cols_to_drop_non_features = ["nominal_head_id", "dependent_id", "nominal_head_lemma", "dependent_lemma", "nominal_head_upos"]
    nominals_df.drop(columns=[col for col in cols_to_drop_non_features if col in nominals_df.columns], inplace=True, errors='ignore')
    print("Dropped non-feature columns (IDs, Lemmas, Nominal Head UPOS).")


    # Encoding Word Position as a Binary Feature (Section 5.4.2)
    # The 'position' column ('before' or 'after') is converted to a numerical binary feature.
    # Mapping: 'before' -> 1, 'after' -> 2, as described in the thesis text.
    if "position" in nominals_df.columns:
        nominals_df['position_numeric'] = nominals_df['position'].apply(lambda x: 1 if x == 'before' else 2)
        nominals_df.drop(columns=["position"], inplace=True, errors='ignore') # Drop the original text column
        print("Encoded 'position' as 'position_numeric'.")
    elif 'position_numeric' not in nominals_df.columns:
        print("Warning: Neither 'position' nor 'position_numeric' column found. Cannot encode position.")
        # The model will proceed without this feature if it's not present


    # One-Hot Encoding Categorical Features (Section 5.4.3 and Table 11)
    # Converts specified categorical columns into multiple binary columns.
    categorical_features_to_ohe = ["dependent_deprel", "dependent_upos", "source_file"]

    # Identify categorical columns present in the DataFrame for OHE
    categorical_features_present_for_ohe = [col for col in categorical_features_to_ohe if col in nominals_df.columns]

    encoded_df = pd.DataFrame(index=nominals_df.index) # Initialize an empty DataFrame for encoded features

    if categorical_features_present_for_ohe:
        # Use OneHotEncoder
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded_features = encoder.fit_transform(nominals_df[categorical_features_present_for_ohe])
        encoded_columns = encoder.get_feature_names_out(categorical_features_present_for_ohe)

        # Create DataFrame for encoded features
        encoded_df = pd.DataFrame(encoded_features, columns=encoded_columns, index=nominals_df.index)
        print(f"One-hot encoded {categorical_features_present_for_ohe}.")
    else:
         print("Warning: No specified categorical features found for one-hot encoding.")

    # Drop the original categorical columns from the main DataFrame, as they are replaced by OHE columns.
    cols_to_drop_original_categorical = [col for col in categorical_features_present_for_ohe if col in nominals_df.columns]
    nominals_df.drop(columns=cols_to_drop_original_categorical, inplace=True, errors='ignore')
    # print(f"Dropped original categorical columns: {cols_to_drop_original_categorical}.") # Less verbose print


    # Generating Interaction Features (Section 5.4.4)
    # Explicit interaction features were excluded from the final LR model
    # due to multicollinearity issues identified in Section 5.8 and Table 7.
    # Therefore, this step is omitted here for the final feature set.


    # Combine all prepared feature columns (numerical, binary, and one-hot encoded).
    # Exclude the target variable 'ezafe_label'.
    feature_cols_in_nominals_df = [col for col in nominals_df.columns if col != 'ezafe_label']

    # Concatenate columns from nominals_df (containing numerical/binary features) and the encoded_df.
    X_combined = pd.concat([nominals_df[feature_cols_in_nominals_df], encoded_df], axis=1)
    y = nominals_df["ezafe_label"] # Target variable

    # Final check to ensure all feature columns are numerical before scaling/fitting.
    non_numeric_cols_in_X = X_combined.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric_cols_in_X:
        print(f"Error: Non-numeric columns still present in feature set X: {non_numeric_cols_in_X}")
        # Print data types to aid debugging
        print(X_combined[non_numeric_cols_in_X].dtypes)
        exit() # Stop execution if non-numeric data persists


    X = X_combined # The complete feature set before scaling and selection
    print(f"\nCombined all features. Shape before scaling/selection: {X.shape}")


    # Define list of actual numerical features present for scaling.
    # These are the columns that are not one-hot encoded and represent quantities or distances.
    actual_numeric_features_list = ["distance", "num_dependents_nominal", "num_dependents_dependent"]
    # Add position_numeric if it exists in the DataFrame
    if 'position_numeric' in X.columns:
        actual_numeric_features_list.append('position_numeric')

    # Ensure the numerical columns actually exist in X before attempting to scale them.
    actual_numeric_features = [col for col in actual_numeric_features_list if col in X.columns]


    # Standardizing Numerical Features using RobustScaler (Section 5.4.5, Table 9 Step 4)
    # RobustScaler is preferred due to its resistance to outliers in linguistic data.
    scaler = RobustScaler()
    if actual_numeric_features: # Apply scaling only if there are numerical features present
        X[actual_numeric_features] = scaler.fit_transform(X[actual_numeric_features])
        print("Standardized numerical features using RobustScaler.")
    else:
        print("No numerical features found to standardize.")


    # --- Feature Importance Analysis and Selection ---
    # Train an initial Logistic Regression model on ALL features in X to get coefficients
    # for calculating feature importance (Section 5.4.6).
    # Using 'liblinear' solver for efficiency on this intermediate step. C=1.0 (default) is used.
    initial_model_for_selection = LogisticRegression(max_iter=MAX_ITER_FOR_IMPORTANCE, class_weight='balanced', solver='liblinear', C=1.0)
    try:
        initial_model_for_selection.fit(X, y)
    except ValueError as e:
         print(f"Error during initial model fit for feature importance: {e}")
         print("Please check if X contains non-numeric values or if there are issues with the data.")
         exit()


    # Compute absolute importance from model coefficients (Section 5.4.6).
    # For binary classification, the coefficients indicate feature influence on the probability.
    if initial_model_for_selection.coef_.shape[0] > 1:
         # Handle potential multi-class if target mapping was unexpected, though LR is binary here.
         print("Warning: Initial model fit resulted in multi-class coefficients. Using coefficients for class 1.")
         coef_values = initial_model_for_selection.coef_[0]
    else:
         coef_values = initial_model_for_selection.coef_.flatten()

    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient Value": coef_values,
    })
    feature_importance["Absolute Importance"] = feature_importance["Coefficient Value"].abs()

    # Select features based on the defined importance threshold (Section 5.4.6, Table 9 Step 8).
    # Features with absolute importance below the threshold are considered less influential and removed.
    selected_features = feature_importance[feature_importance["Absolute Importance"] > IMPORTANCE_THRESHOLD]["Feature"].tolist()

    # Filter the feature set X to keep only the selected features.
    X_final = X[selected_features]

    print(f"Selected {len(selected_features)} features based on importance threshold {IMPORTANCE_THRESHOLD}.")
    print(f"Shape of X after feature selection: {X_final.shape}")

    # Save the feature importance results for documentation (Section 5.4.6).
    # Sorting by importance before saving makes the output more interpretable.
    feature_importance_sorted = feature_importance.sort_values(by="Absolute Importance", ascending=False)
    try:
        feature_importance_sorted.to_csv(FEATURE_IMPORTANCE_CSV, index=False)
        print(f"Feature importance for all features saved to '{FEATURE_IMPORTANCE_CSV}'")
    except Exception as e:
         print(f"Error saving feature importance CSV: {e}")


    # Save the final feature set (X_final) and target (y) to CSVs.
    # These files serve as the direct input for the lr_train_evaluate.py script.
    try:
        X_final.to_csv(OUTPUT_FEATURES_CSV, index=False)
        y.to_csv(OUTPUT_TARGET_CSV, index=False)
        print(f"\nFinal feature set and target saved to '{OUTPUT_FEATURES_CSV}' and '{OUTPUT_TARGET_CSV}'.")
    except Exception as e:
         print(f"Error saving final feature/target CSVs: {e}")


# --- Configuration for Initial Model Fit (Needed for MAX_ITER_FOR_IMPORTANCE) ---
# This configuration is specifically for the temporary model used to calculate initial feature importance.
# It's placed after the main block for organizational clarity, but needed for MAX_ITER_FOR_IMPORTANCE definition.
MAX_ITER_FOR_IMPORTANCE = 500 # Max iterations for the initial model to calculate coefficients
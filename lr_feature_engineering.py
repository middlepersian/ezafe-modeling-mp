# Filename: lr_feature_engineering.py
# Author: Raha Musavi
# Matrikelnummer: 08022255354
# Thesis: Modeling the Presence and the Classification of Ezafe in Middle Persian Nominal Phrases 
# Date: 01 May 2025


"""
This script loads the cleaned nominal phrase data CSV, performs feature engineering
specific to the Logistic Regression model, including encoding categorical features,
standardizing numerical features, and calculating feature importance. It then selects
the most important features based on a defined threshold. The final feature set
and the target variable are saved to separate CSV files, which serve as input
for the model training script. It also calculates and saves the correlation matrix
between features and the target for analysis.

Corresponds to Feature Engineering (Section 5.4), Scaling (Section 5.4.5),
Feature Importance Analysis (Section 5.4.6), Feature Selection, and Correlation Analysis
(Section 5.10.3) steps in Chapter 5 of the thesis.
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
INPUT_CSV_PATH = "np_inputs.csv"
OUTPUT_FEATURES_CSV = "lr_final_features.csv"
OUTPUT_TARGET_CSV = "lr_target.csv"
FEATURE_IMPORTANCE_CSV = "LR_feature_importance_all_features.csv"
CORRELATION_MATRIX_CSV = "feature_correlation_matrix.csv" # For correlation matrix output

# Define the threshold for feature selection based on absolute importance (Section 5.4.6, Table 9 Step 8)
# Features with absolute coefficient value below this threshold are removed.
IMPORTANCE_THRESHOLD = 0.15

# Configuration for the temporary initial model fit used only for feature importance calculation
MAX_ITER_FOR_IMPORTANCE = 500 # Max iterations for the initial model to calculate coefficients

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

    # Encoding Word Position as a Binary Feature (Section 5.4.2)
    # The 'position' column ('before' or 'after') is converted to a numerical binary feature.
    # Mapping: 'before' -> 1, 'after' -> 2, as described in the thesis text.
    # Do this early as it's used in correlation and the main feature set.
    if "position" in nominals_df.columns:
        nominals_df['position_numeric'] = nominals_df['position'].apply(lambda x: 1 if x == 'before' else 2)
        nominals_df.drop(columns=["position"], inplace=True, errors='ignore') # Drop the original text column
        print("Encoded 'position' as 'position_numeric'.")
    elif 'position_numeric' not in nominals_df.columns:
        print("Warning: Neither 'position' nor 'position_numeric' column found. Cannot encode position.")
        # The model will proceed without this feature if it's not present


    # --- Compute and Save Correlation Matrix (Section 5.10.3, Table 10) ---
    # Calculate correlation of all relevant features with the target variable.
    # This needs features *before* the main selection step.
    # The OHE needs to be applied for categorical features for correlation calculation.

    # Define features needed for correlation calculation (including original categories for OHE)
    corr_features_list = ["distance", "num_dependents_nominal", "num_dependents_dependent",
                          "is_verbal", "nominal_head_upos", # Keep original nominal head upos for this table
                          "dependent_upos", "dependent_deprel", "source_file"] # Keep original dependent features for OHE

    # Add position_numeric if it was created
    if 'position_numeric' in nominals_df.columns:
         corr_features_list.append('position_numeric')

    # Filter to keep only columns that exist in the dataframe
    corr_cols_present = [col for col in corr_features_list if col in nominals_df.columns]

    # Create a temporary DataFrame for correlation calculation
    temp_df_for_corr = nominals_df[corr_cols_present + ['ezafe_label']].copy() # Include target and copy

    # One-Hot Encode the categorical features for the correlation matrix table (Section 5.10.3, Table 10 lists OHE features)
    corr_categorical_features_to_ohe = ["dependent_deprel", "dependent_upos", "source_file", "nominal_head_upos"] # Include head upos for this table
    corr_categorical_features_present = [col for col in corr_categorical_features_to_ohe if col in temp_df_for_corr.columns]

    if corr_categorical_features_present:
        corr_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        corr_encoded_features = corr_encoder.fit_transform(temp_df_for_corr[corr_categorical_features_present])
        corr_encoded_columns = corr_encoder.get_feature_names_out(corr_categorical_features_present)
        corr_encoded_df = pd.DataFrame(corr_encoded_features, columns=corr_encoded_columns, index=temp_df_for_corr.index)
        # Drop original categorical columns from temp df and concatenate OHE ones
        temp_df_for_corr = pd.concat([temp_df_for_corr.drop(columns=corr_categorical_features_present), corr_encoded_df], axis=1)
        # print(f"One-hot encoded {corr_categorical_features_present} for correlation calculation.") # Less verbose

    # Ensure all columns in temp_df_for_corr used in correlation are numeric
    non_numeric_corr_cols = temp_df_for_corr.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric_corr_cols:
        print(f"Warning: Non-numeric columns found in temporary correlation df: {non_numeric_corr_cols}")
        # Attempt to convert them to numeric if possible, otherwise drop them for corr calculation
        for col in non_numeric_corr_cols:
            try:
                temp_df_for_corr[col] = pd.to_numeric(temp_df_for_corr[col])
            except (ValueError, TypeError):
                print(f"Could not convert column '{col}' to numeric for correlation. Dropping.")
                temp_df_for_corr.drop(columns=[col], inplace=True)


    # Calculate correlation matrix for the relevant columns in the temporary dataframe
    if not temp_df_for_corr.empty and 'ezafe_label' in temp_df_for_corr.columns:
        correlation_matrix = temp_df_for_corr.corr()

        # Extract correlation with the target variable 'ezafe_label' and sort
        correlation_with_target = correlation_matrix['ezafe_label'].sort_values(ascending=False)

        # Save to CSV (Section 5.10.3 references Table 10 which shows this)
        correlation_with_target_df = correlation_with_target.reset_index()
        correlation_with_target_df.columns = ["Feature", "Correlation with ezafe_label"]
        try:
            correlation_with_target_df.to_csv(CORRELATION_MATRIX_CSV, index=False)
            print(f"Correlation matrix with target saved to '{CORRELATION_MATRIX_CSV}'.")
        except Exception as e:
            print(f"Error saving correlation matrix CSV: {e}")
    else:
        print("Warning: Temporary DataFrame is empty or missing 'ezafe_label' for correlation calculation.")

    del temp_df_for_corr # Clean up temporary DataFrame


    # --- Continue with Feature Engineering for the Model ---
    # Drop columns not used as features for LR (IDs, Lemmata, Nominal Head UPOS) for the main feature set X
    # Nominal Head UPOS is dropped as it did not significantly contribute to the model (Section 5.4.1).
    cols_to_drop_non_features_main = ["nominal_head_id", "dependent_id", "nominal_head_lemma", "dependent_lemma", "nominal_head_upos"]
    nominals_df.drop(columns=[col for col in cols_to_drop_non_features_main if col in nominals_df.columns], inplace=True, errors='ignore')
    print("Dropped non-feature columns for model input (IDs, Lemmata, Nominal Head UPOS).")


    # One-Hot Encoding Categorical Features for the Model (Section 5.4.3 and Table 11)
    # Converts specified categorical columns into multiple binary columns.
    # Note: nominal_head_upos is NOT included here, only for the correlation table above.
    categorical_features_to_ohe_main = ["dependent_deprel", "dependent_upos", "source_file"]

    # Identify categorical columns present in the DataFrame for OHE for the model
    categorical_features_present_for_ohe_main = [col for col in categorical_features_to_ohe_main if col in nominals_df.columns]

    encoded_df = pd.DataFrame(index=nominals_df.index) # Initialize an empty DataFrame for encoded features

    if categorical_features_present_for_ohe_main:
        # Use OneHotEncoder
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded_features = encoder.fit_transform(nominals_df[categorical_features_present_for_ohe_main])
        encoded_columns = encoder.get_feature_names_out(categorical_features_present_for_ohe_main)

        # Create DataFrame for encoded features
        encoded_df = pd.DataFrame(encoded_features, columns=encoded_columns, index=nominals_df.index)
        print(f"One-hot encoded {categorical_features_present_for_ohe_main} for model input.")
    else:
         print("Warning: No specified categorical features found for one-hot encoding for model input.")


    # Drop the original categorical columns from the main DataFrame, as they are replaced by OHE columns.
    cols_to_drop_original_categorical_main = [col for col in categorical_features_present_for_ohe_main if col in nominals_df.columns]
    nominals_df.drop(columns=cols_to_drop_original_categorical_main, inplace=True, errors='ignore')


    # Generating Interaction Features (Section 5.4.4)
    # Explicit interaction features were excluded from the final LR model
    # due to multicollinearity issues identified in Section 5.8 and Table 7.
    # Therefore, this step is omitted here for the final feature set used by the model.


    # Define list of numerical features present for scaling.
    # These are the columns that are not one-hot encoded or binary (is_verbal)
    actual_numeric_features_list = ["distance", "num_dependents_nominal", "num_dependents_dependent"]
    # Add position_numeric if it exists in the DataFrame
    if 'position_numeric' in nominals_df.columns:
        actual_numeric_features_list.append('position_numeric')

    # Ensure the numerical columns actually exist in nominals_df
    actual_numeric_features = [col for col in actual_numeric_features_list if col in nominals_df.columns]

    # Define list of binary features present
    actual_binary_features = ["is_verbal"]
    actual_binary_features = [col for col in actual_binary_features if col in nominals_df.columns]


    # Combine all prepared feature columns (numerical, binary, and one-hot encoded).
    # Exclude the target variable 'ezafe_label'.
    feature_cols_in_nominals_df = actual_numeric_features + actual_binary_features # Numerical and binary features from nominals_df

    # Concatenate columns from nominals_df and the encoded_df.
    X_combined = pd.concat([nominals_df[feature_cols_in_nominals_df], encoded_df], axis=1)
    y = nominals_df["ezafe_label"] # Target variable


    # Final check to ensure all feature columns in X_combined are numerical.
    non_numeric_cols_in_X = X_combined.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric_cols_in_X:
        print(f"Error: Non-numeric columns still present in feature set X before scaling: {non_numeric_cols_in_X}")
        # Print data types to aid debugging
        print(X_combined[non_numeric_cols_in_X].dtypes)
        exit() # Stop execution if non-numeric data persists


    X = X_combined # The complete feature set before scaling and selection
    print(f"\nCombined all features. Shape before scaling/selection: {X.shape}")


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
    # coef_ has shape (1, n_features) for binary classification.
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

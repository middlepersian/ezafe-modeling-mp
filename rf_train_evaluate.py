# src/rf_train_evaluate.py
# Author: Raha Musavi
# Matrikelnummer: 08022255354
# Thesis: Modeling the Presence and the Classification of Ezafe in Middle Persian Nominal Phrases
# Date: 01 May 2025

"""
This script trains and evaluates the Random Forest classification model
for classifying ezafe constructions based on presence and head position,
corresponding to the "Combined Labels Model" described in Chapter 6 of the thesis.

It loads the prepared feature dataset, applies necessary encoding (Label Encoding
and TF-IDF vectorization), handles class imbalance using Random Oversampling,
splits the data into training and testing sets, trains the Random Forest model
with optimized hyperparameters, evaluates its performance using standard
classification metrics and a confusion matrix, and saves the trained model
and the confusion matrix plot.

This specific configuration corresponds to the model trained *before*
the VIF-based feature filtering step discussed in Section 6.8 of the thesis.

Note on Reproducibility:
The results obtained by running this script may show minor variations compared
to the exact metrics reported in Chapter 6 (e.g., Table 19, Table 20) of the
thesis. While efforts were made to fix random states and use consistent data
processing, slight differences in library versions (e.g., scikit-learn, imblearn,
pandas) or underlying system environments can influence floating-point
calculations or the specific execution paths in algorithms, leading to minor
divergences in outcomes. An observed slight improvement in performance
upon re-running this code, potentially relative to the thesis results,
could be attributable to updates in the Python ecosystem around the time
the model was finalized (e.g., Python updates around April 8, 2025).

Input file: data/rf_features.csv (containing all extracted features and the 'combined_label')
Output files:
    - models/final_random_forest_model_previf.pkl (the trained model object)
    - figures/rf_confusion_matrix_previf.png (heatmap visualization of the confusion matrix)
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import gc

# --- Main execution for rf_train_evaluate.py ---
if __name__ == "__main__":
    """
    Main execution block for the Random Forest training and evaluation script.

    Loads data, prepares features, handles imbalance, splits data, trains,
    evaluates, and saves the model and evaluation artifacts.
    """
    # Get the directory where the script is located
    script_dir = os.path.dirname(__file__)
    # Construct paths to data, models, figures folders relative to the script's location
    data_dir = os.path.join(script_dir, '..', 'data')
    models_dir = os.path.join(script_dir, '..', 'models')
    results_dir = os.path.join(script_dir, '..', 'results') # Results directory is mentioned in README
    figures_dir = os.path.join(script_dir, '..', 'figures')

    # Create necessary directories if they do not exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True) # Create results directory
    os.makedirs(figures_dir, exist_ok=True)

    print("--- Step 3: RF Model Training and Evaluation (rf_train_evaluate.py) - Pre-VIF Model ---")

    # Define the path to the prepared features file
    features_file = os.path.join(data_dir, "rf_features.csv")

    # Check if the input file exists before proceeding
    if not os.path.exists(features_file):
        print(f"Error: Features file '{features_file}' not found. Ensure rf_prepare_features.py was run successfully.")
        exit() # Terminate script if input file is missing

    # Load the prepared dataset into a pandas DataFrame.
    df = pd.read_csv(features_file)

    # Extract the target variable ('combined_label') and the features.
    # The 'combined_label' column contains the merged ezafe presence and head position information.
    if 'combined_label' not in df.columns:
        print("Error: 'combined_label' column not found in the features file. Ensure rf_prepare_features.py created it.")
        exit()

    # Encode the combined target label into numerical classes (0, 1, 2, 3).
    # This LabelEncoder will be used to map the integer predictions back to class names.
    le_combined = LabelEncoder()
    y = le_combined.fit_transform(df['combined_label'])

    # Define feature groups based on the original notebook's preprocessing (Cell 2)
    # and the feature list in the thesis tables (Tables 12-17), excluding IDs and original targets.
    # This corresponds to the feature set used for the "Combined Labels Model" (Pre-VIF).
    id_cols_to_drop = ['nominal_head_id', 'nominal_head_form', 'dependent_id', 'dependent_form'] # Columns not used as model features
    original_target_cols = ['ezafe_label', 'position', 'combined_label'] # Original target columns

    # Categorical features that will be Label Encoded
    le_cols = ['nominal_head_upos', 'nominal_head_deprel', 'head_number',
               'dependent_upos', 'dependent_deprel', 'source_file']

    # The column containing the dependency pattern string for TF-IDF vectorization
    tfidf_col = 'np_deprel_pattern'

    # Identify numerical columns by excluding known non-numerical features and targets
    all_cols = df.columns.tolist()
    cols_to_exclude_from_numeric_check = id_cols_to_drop + original_target_cols + le_cols + [tfidf_col]
    cols_to_exclude_from_numeric_check = [col for col in cols_to_exclude_from_numeric_check if col in all_cols]
    numeric_cols = [col for col in all_cols if col not in cols_to_exclude_from_numeric_check]


    # --- Apply Feature Transformations ---

    # Create a copy of the DataFrame for processing to avoid modifying the original loaded data
    df_processed = df.copy()

    # Apply Label Encoding to specified categorical features and convert to sparse columns.
    le_cols_present = [col for col in le_cols if col in df_processed.columns]
    X_le_list = []
    for col in le_cols_present:
         # Fill any potential missing values before encoding
         df_processed[col] = df_processed[col].astype(str).fillna("unknown_le_placeholder")
         # Fit and transform using a temporary LabelEncoder instance
         le_output = LabelEncoder().fit_transform(df_processed[col])
         # Convert the numerical output to a sparse column matrix for efficient stacking
         X_le_list.append(csr_matrix(le_output).transpose())

    # Horizontally stack all sparse Label Encoded columns. Handle the case where there are no such columns.
    X_le = hstack(X_le_list) if X_le_list else csr_matrix((df_processed.shape[0], 0))


    # Select numerical columns, fill potential missing values, and convert to a sparse matrix.
    numeric_cols_present = [col for col in numeric_cols if col in df_processed.columns]
    X_num_dense = df_processed[numeric_cols_present].apply(pd.to_numeric, errors='coerce').fillna(-1) # Fill NaNs with -1
    X_num = csr_matrix(X_num_dense.values) # Convert dense numpy array to sparse matrix


    # Apply TF-IDF vectorization to the dependency pattern string feature.
    if tfidf_col in df_processed.columns:
        # Fill any potential missing values with an empty string before vectorization
        deprel_patterns = df_processed[tfidf_col].astype(str).fillna("")
        # Initialize and fit/transform the TF-IDF vectorizer using the same parameters as the original notebook.
        tfidf_vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\S+', max_features=1000)
        X_tfidf = tfidf_vectorizer.fit_transform(deprel_patterns)
    else:
        print(f"Warning: TF-IDF column '{tfidf_col}' not found in the DataFrame.")
        X_tfidf = csr_matrix((df_processed.shape[0], 0)) # Create an empty sparse matrix if the column is missing


    # Combine all the transformed feature matrices (numerical, LE categoricals, TF-IDF pattern)
    # into the final feature matrix X for model training.
    X_final = hstack([X_num, X_le, X_tfidf])


    # Clean up temporary dataframes and variables to free memory before training
    del df, df_processed, X_num_dense, X_le_list, deprel_patterns
    gc.collect() # Explicitly call garbage collection

    print(f"\nFinal feature shape after encoding: {X_final.shape}")
    print(f"Target shape: {y.shape}")


    # --- Handle Class Imbalance ---
    # Apply Random Oversampling (ROS) to balance the class distribution in the training data.
    # This helps prevent the model from being biased towards the majority classes during training.
    ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_final, y)

    print(f"\nResampled data shape: {X_resampled.shape}")
    print(f"Resampled target distribution:\\n{pd.Series(y_resampled).value_counts().sort_index()}")

    # --- Split Data ---
    # Split the resampled data into training and testing sets (80/20 split).
    # Use stratified sampling to ensure the balanced class distribution is maintained in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

    # Print the class distribution in the stratified test set for verification.
    print("\nClass distribution in the STRATIFIED Test Set:")
    unique, counts = np.unique(y_test, return_counts=True)
    print(dict(zip(le_combined.inverse_transform(unique), counts)))


    # --- Model Training ---
    # Define the optimal hyperparameters for the Random Forest model as determined
    # through hyperparameter tuning (Section 6.6 of the thesis).
    best_params = {'n_estimators': 200, 'min_samples_split': 2, 'max_depth': None, 'class_weight': 'balanced'}

    print(f"\nInitializing the model with parameters: {best_params}")
    # Initialize the RandomForestClassifier with the specified parameters and random state for reproducibility.
    rf_model = RandomForestClassifier(random_state=42, **best_params)

    print("Fitting the model to the training data...")
    # Train the model on the resampled training data.
    rf_model.fit(X_train, y_train)
    print("Model fitting completed.")

    # --- Model Evaluation ---
    # Predict the target labels on the test set using the trained model.
    y_pred = rf_model.predict(X_test)
    # Predict probabilities for the test set.
    y_pred_proba = rf_model.predict_proba(X_test)

    print("\n--- Model Evaluation Results ---")
    # Print the classification report showing key metrics (precision, recall, F1-score) for each class.
    print("Classification Report:")
    # Use digits=4 for precision consistency with thesis tables (Table 19, Table 20).
    print(classification_report(y_test, y_pred, zero_division='warn', digits=4))

    # Calculate and print the overall accuracy of the model on the test set.
    overall_accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {overall_accuracy:.4f}")

    # Define the mapping from integer labels (used by the model) back to human-readable class names.
    encoded_class_integers = rf_model.classes_
    decoded_class_strings = le_combined.inverse_transform(encoded_class_integers)

    # Map the decoded string labels to descriptive names for the confusion matrix plot.
    human_readable_label_mapping = {
        "0_1": "No Ezafe & Head Initial",
        "0_2": "No Ezafe & Head Final",
        "1_1": "With Ezafe & Head Initial",
        "1_2": "With Ezafe & Head Final"
    }
    readable_labels_ordered = [human_readable_label_mapping.get(label_str, label_str) for label_str in decoded_class_strings]

    # Calculate the confusion matrix, comparing actual test labels to predicted labels.
    cm = confusion_matrix(y_test, y_pred, labels=encoded_class_integers)

    print("\nConfusion Matrix (Raw Data):")
    print(cm)

    # Plot the confusion matrix as a heatmap for clear visualization of performance across classes.
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="cividis",
                xticklabels=readable_labels_ordered, yticklabels=readable_labels_ordered)
    plt.xlabel('Predicted Class')
    plt.xticks(rotation=0, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.ylabel('Actual Class')
    plt.title('Confusion Matrix for Random Forest Model (Pre-VIF)')
    plt.tight_layout()

    # Save the confusion matrix plot to the figures directory.
    output_figure_path = os.path.join(figures_dir, 'rf_confusion_matrix_previf.png')
    plt.savefig(output_figure_path)
    print(f"Confusion matrix plot saved to {output_figure_path}")
    plt.show() # Display the plot after saving

    # --- Save Trained Model ---
    # Save the trained Random Forest model object to a file.
    # This allows the trained model to be loaded and used later for predictions without retraining.
    model_output_path = os.path.join(models_dir, 'final_random_forest_model_previf.pkl') # Consistent naming
    joblib.dump(rf_model, model_output_path)
    print(f"\nTrained Random Forest model saved to {model_output_path}")

    # Note on further results: Feature importance results (Gini, Permutation, SHAP)
    # and detailed error analysis (Section 6.10.6, Table 21) are generated and
    # analyzed in separate notebooks/scripts and can be saved to the 'results/' directory.


    print("\n--- RF Model Training and Evaluation Complete ---")
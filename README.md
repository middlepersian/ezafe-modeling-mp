# Code for Modeling Ezafe in Middle Persian Nominal Phrases

This repository contains the code and analysis scripts developed for the Master's thesis "Modeling the Presence and the Classification of Ezafe in Middle Persian Nominal Phrases".

**Author:** Raha Musavi
**Matrikelnummer:** 108022255354
**Thesis Submission Date:** 01 May 2025

## Project Description

This project applies machine learning techniques to analyze and model the grammatical phenomenon of ezafe in an annotated corpus of Middle Persian nominal phrases. The primary objectives were to identify the linguistic and structural features influencing the presence of ezafe and to classify different types of ezafe constructions. This repository provides the code for the two main machine learning models used in the thesis:

*   **Logistic Regression:** Used for the binary classification task of identifying Ezafe presence (Chapter 5).
*   **Random Forest:** Used for the multi-class classification task of classifying Ezafe constructions based on presence and head position (Chapter 6).

## Repository Structure

The repository is organized as follows:

*   `src/`: Contains the core Python scripts for the data processing pipeline, including scripts specific to each model's feature engineering and training.
*   `notebooks/`: Contains Jupyter Notebooks used for detailed analysis and visualization of the results for each model.
*   `data/`: Intended for storing processed data files (`.csv`) that serve as input for the analysis pipeline steps.
*   `results/`: Intended for storing output files from the analysis scripts, such as correlation matrices and feature importance results.
*   `models/`: Intended for storing trained model files (`.pkl`).
*   `figures/`: Intended for storing saved plots generated during the analysis.
*   `.gitignore`: Specifies files and directories that Git should ignore (e.g., virtual environments, large data files not included).
*   `requirements.txt`: Lists the necessary Python libraries to run the code.
*   `LICENSE`: Specifies the project's license.
*   `README.md`: This file.

## Setup and Requirements

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/RahaaMusavi/ezafe-modeling-mp.git
    cd ezafe-modeling-mp
    ```
2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```
3.  **Install dependencies:** Install the required Python libraries using pip and the `requirements.txt` file (you will need to create this file if it's not already in the repo based on your environment).
    ```bash
    pip install -r requirements.txt
    ```

## Data

The raw data used in this project is the annotated Zoroastrian Middle Persian corpus developed by the MPCD project at Ruhr-Universität Bochum. This corpus is not publicly available in this repository due to its project status.

*   The raw CoNLL-U data can be obtained from the MPCD project website: [https://www.mpcorpus.org/corpus/](https://www.mpcorpus.org/corpus/). Specific access instructions or requirements may apply; please refer to the website or contact the project directly for details (Prof. Dr. Kianoosh Rezania).

## How to Run the Code

The analysis pipeline involves multiple sequential Python scripts located in the `src/` directory. Ensure your Python environment is activated and you are in the root directory of the repository.

1.  **Initial Preprocessing (Common to both models):**
    ```bash
    python src/lr_preprocess_data.py
    ```
    *   This script reads the raw `.conllu` files, cleans and extracts initial features, and saves the result to `data/lr_np_inputs.csv`.
    *   **Note:** If `data/n.csv` is already included in the repository and you intend to use it, you can skip this step.
    
2.  **Logistic Regression Pipeline:**
    *   **Feature Engineering and Selection:**
        ```bash
        python src/lr_feature_engineering.py
        
        ```
3.  **Random Forest Pipeline:**
    *   **Feature Extraction (RF Specific):**
        ```bash
        python src/rf_feature_extraction.py
        ```
        *   This script loads `data/rf_np_inputs.csv`, performs RF-specific feature extraction (including structural/complexity features, creating the combined target), and saves the RF feature set (`data/rf_features.csv`) and combined target (`data/rf_combined_target.csv`).
        *   **Note on `is_verbal` Feature:** Due to the filtering of clausal dependents (Section 6.3.1) to focus on nominal phrases, the `is_verbal` feature in the extracted dataset is expected to have no instances marked as true (1). This feature is intentionally included to verify the effectiveness of the filtering process and to ensure that the machine learning pipeline correctly handles features with minimal or zero variance.
    *   **Model Training and Evaluation:**
        ```bash
        python src/lr_train_evaluate.py
        ```
        *   This script loads the data from `data/lr_final_features.csv` and `data/lr_target.csv`, handles class imbalance, splits the data, performs hyperparameter tuning, trains the final Logistic Regression model, evaluates it, and saves the trained model to `models/final_logistic_regression_model.pkl`.

3.  **Random Forest Pipeline:**
    *   **Feature Extraction (RF Specific):**
        ```bash
        python src/rf_feature_extraction.py
        ```
        *   This script loads `data/nominal_features_cleaned.csv`, performs RF-specific feature extraction (including structural/complexity features, creating the combined target), and saves the RF feature set (`data/rf_features.csv`) and combined target (`data/rf_combined_target.csv`).
    *   **Model Training and Evaluation:**
        ```bash
        python src/rf_train_evaluate.py
        ```
        *   This script loads the data from `data/rf_features.csv` and `data/rf_combined_target.csv`, handles class imbalance, splits the data, performs hyperparameter tuning, trains the final Random Forest model, evaluates it, and saves the trained model to `models/final_random_forest_model.pkl`, as well as RF importance results (`results/`).

## How to View the Analysis

Detailed analysis and visualization for each model are presented in separate Jupyter Notebooks located in the `notebooks/` directory.

1.  **Navigate to the notebook directory:**
    ```bash
    cd notebooks
    ```
2.  **Launch Jupyter Notebook or JupyterLab:**
    ```bash
    jupyter notebook
    # or
    jupyter lab
    ```
3.  Open the desired notebook (`lr_analysis_notebook.ipynb` or `rf_analysis_notebook.ipynb`). The notebooks contain embedded outputs from a previous run. You can execute the cells yourself to reproduce the results (assuming the previous scripts ran successfully and generated the necessary input files for that model).

## Notes on Model Configuration and Results Discrepancies

The code in this repository implements the final methodologies and model configurations for both the Logistic Regression (Chapter 5) and Random Forest (Chapter 6) analyses as presented in the thesis.

During the iterative development process of the Logistic Regression model described in the thesis, the feature sets for each model evolved. Notably, the `source_file` feature was introduced later in the analysis phase to investigate text-specific variations and was included in the final model configurations (as reflected in Table 11 and Table 20, and this repository's code).

Consequently, some figures and discussions in the thesis (prior to the final analysis sections) which depict results obtained from earlier model iterations (with different feature sets or potentially with different hyperparameter tuning outcomes) may show minor visual or numerical differences when compared to the outputs generated by this notebook for the final model configurations.

The results presented in the output cells of the notebooks are consistent with the values reported in the respective tables of the thesis (Table 11 for LR, Table 20 for RF) and reflect the performance of the models trained with the final feature sets including the `source_file` feature. 

### Tools and Assistance

Some comments and docstrings within the Python scripts and Jupyter Notebooks were generated with the assistance of an AI language model (OpenAI). All AI-generated content has been thoroughly reviewed, checked for accuracy, and approved by the author to ensure it correctly reflects the implemented methodology and analysis. The core logic and design of the code are the author's original work.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact: seyyedehfatemeh.musavi@ruhr-uni-bochum.de

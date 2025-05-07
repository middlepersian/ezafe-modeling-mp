# Filename: lr_preprocess_data.py

# Author: Raha Musavi
# Matrikelnummer: 08022255354
# Thesis: Modeling the Presence and the Classification of Ezafe in Middle Persian Nominal Phrases 
# Date: 01 May 2025

"""
This script processes raw linguistic data in CoNLL-U format.
It reads files from a specified directory, parses the data, and extracts
relevant features for nominal phrases and their dependents. The script
performs initial cleaning steps, including standardizing dependency relations
and handling token ID formats, before saving the processed data to a CSV file.
This output file serves as the input for subsequent feature engineering and
model training stages.

Corresponds to the data loading and initial preprocessing steps described in
Chapter 4 and Section 5.3 of the thesis. One feature was added during the error analysis and feature evaluation step, namely source_file.
It showed that the results from the model are fairly stable and improve with the consideration of the source_file differences.
"""

import os
import pandas as pd
from conllu import parse_incr # Requires: pip install conllu
import warnings

# Suppress warnings that may arise during processing.
warnings.filterwarnings("ignore")

# --- Configuration ---
# Define the folder containing the raw CoNLL-U files.
# This path must be updated to reflect the actual location of the data files.
CONLLU_FOLDER = r'C:\Users\rahaa\Dropbox\MPCD\conllus_with_erros' # <--- *** UPDATE THIS PATH ***

# Define the output path for the cleaned nominal features CSV file.
OUTPUT_CSV_PATH = "np_inputs.csv"

# Define a dictionary for standardizing dependency labels to correct variations (Section 5.3.5, Table B4 footnote)
standardized_deprels = {
    "nm": "nmod",
    "adjmod": "amod",
    "al:relcl": "acl:relcl",
    "acl:recl": "acl:relcl",
    "advcl.": "advcl" # Correcting a specific typo identified (Table B4 footnote)
}

# Define dependency relations considered as stopwords to filter out dependents (Section 5.3.1)
stopwords_deprels = {'punct', 'dep', 'reparandum'}

# --- Helper Functions ---

def standardize_deprel(deprel):
    """Standardizes a given dependency relation label based on a predefined dictionary."""
    return standardized_deprels.get(deprel, deprel)

def convert_tuple_id_to_float(id_tuple):
    """
    Converts token IDs from integer or tuple formats (used in Enhanced Dependencies)
    into a standardized float format for consistent processing (Section 5.3.3).
    Returns None if the ID format is unexpected or cannot be converted.
    """
    if isinstance(id_tuple, tuple) and len(id_tuple) == 3:
        integer_part = id_tuple[0]
        decimal_part = id_tuple[2]
        try:
            # Attempt conversion, handling potential non-digit parts gracefully
            int_part = int(integer_part) if str(integer_part).isdigit() else 0 # Assuming '-' or non-digits in parts can be treated as 0 for structure
            dec_part = int(decimal_part) if str(decimal_part).isdigit() else 0
            if dec_part == 0:
                return float(int_part)
            else:
                # Combine integer and decimal parts as a float (e.g., 3.1)
                id_num = f"{int_part}.{dec_part}"
                return float(id_num)
        except (ValueError, TypeError):
            return None # Indicate conversion failure
    elif isinstance(id_tuple, (int, float)):
         return float(id_tuple)
    else:
         return None # Indicate unexpected format

def adjust_numerical_values(df):
    """
    Increments specific numerical columns by 1 if they contain zero values.
    This step is performed for features that may require non-zero values
    for certain transformations or model inputs, contributing to stability
    (Section 3.1.1, 5.3.2).
    """
    # Numerical columns identified as needing potential zero adjustment
    cols_to_check_for_zero = ["distance", "num_dependents_nominal", "num_dependents_dependent"]

    for col in cols_to_check_for_zero:
        if col in df.columns and (df[col] == 0).any():
            df[col] = df[col] + 1 # Shift all values in this column by 1
    return df

def process_nominal_heads(token_dicts_list, source_file):
    """
    Extracts and structures linguistic features for nominal head - dependent pairs
    within a given sentence's token list. Identifies nominal heads and extracts
    features for their immediate dependents, preparing data for analysis.
    (Based on feature list in Section 5.3)

    Args:
        token_dicts_list (list): A list of dictionaries, where each dictionary
                                 represents a token with parsed CoNLL-U fields.
        source_file (str): The name of the source file the tokens came from.

    Returns:
        list: A list of dictionaries, each representing a head-dependent pair
              with extracted features.
    """
    nominal_data = []
    potential_ezafe_markers = {'ī'} # The lemma for the ezafe token (Chapter 4.1)

    # Create a dictionary for quick token lookup by ID
    token_dict_by_id = {token['id']: token for token in token_dicts_list if token['id'] is not None}

    for token in token_dicts_list:
        # Filter for nominal heads based on UPOS tags (NOUN, PROPN, PRON) and valid ID (Section 5.3)
        if token.get('upos') in {'NOUN', 'PROPN', 'PRON'} and token.get('id') is not None:
            head_id = token['id']
            head_token = token # Reference to the head token dictionary

            # Find immediate dependents with valid head IDs pointing to this token (Section 5.3)
            dependents = [
                t for t in token_dicts_list
                if t.get('head') == head_id and t.get('id') is not None and t['id'] != head_id
            ]
            num_dependents = len(dependents) # Number of immediate dependents of the head

            for dependent in dependents:
                dependent_id = dependent['id']
                dependent_token = dependent # Reference to the dependent token dictionary

                # Calculate distance between head and dependent (Section 5.3)
                distance = abs(dependent_id - head_id)

                # Determine position of the dependent relative to the head (Section 5.3)
                position = 'before' if dependent_id < head_id else 'after'

                # Calculate number of immediate dependents of the dependent (Section 5.3)
                num_dependents_of_dependent = sum(
                    1 for t in token_dicts_list
                    if t.get('head') == dependent_id and t.get('id') is not None and t['id'] != dependent_id
                )

                # Check for ezafe presence. Based on the interpretation from the notebook code
                # and thesis discussion (Modifier + ezafe + Head structure, Section 1.1 examples 2, 5),
                # this checks if an 'ī' token's head is the *dependent* (modifier).
                has_ezafe = any(
                    ezafe_token.get('lemma') in potential_ezafe_markers and
                    ezafe_token.get('head') == dependent_id and
                    ezafe_token.get('id') is not None
                    for ezafe_token in token_dicts_list
                )

                # Check if the dependent (modifier) is a verbal element (VERB or AUX) (Section 5.3)
                is_verbal = int(dependent_token.get('upos') in {'VERB', 'AUX'})

                # Append extracted features for this head-dependent pair
                # ezafe_label: 1 for Ezafe Present, 2 for Ezafe Absent, to match the confusion matrix labels (Section 5.9.1, Cell 13)
                ezafe_label = 1 if has_ezafe else 2

                nominal_data.append({
                    'nominal_head_id': head_id, # Kept for potential manual error analysis/interpretation
                    'nominal_head_lemma': head_token['lemma'], # Kept for potential manual error analysis/interpretation
                    'nominal_head_upos': head_token['upos'], # Kept for potential manual error analysis/interpretation
                    'dependent_id': dependent_id, # Kept for potential manual error analysis/interpretation
                    'dependent_lemma': dependent_token['lemma'], # Kept for potential manual error analysis/interpretation
                    'dependent_upos': dependent_token['upos'],
                    'dependent_deprel': dependent_token['deprel'],
                    'distance': distance,
                    'position': position,
                    'num_dependents_nominal': num_dependents,
                    'num_dependents_dependent': num_dependents_of_dependent,
                    'ezafe_label': ezafe_label, # Target variable
                    'is_verbal': is_verbal, # Feature
                    'source_file': source_file # Feature added after step 8 (Table 9), results in Table 11.
                })

    return nominal_data


def process_conllu_file(file_path):
    """
    Reads and processes a single CoNLL-U file.
    Parses the file into sentences and tokens, standardizes dependency labels,
    filters stopwords, and extracts nominal phrase features from the cleaned tokens.

    Args:
        file_path (str): The full path to the CoNLL-U file.

    Returns:
        list: A list of dictionaries, each representing a head-dependent pair
              with extracted features from this file.
    """
    file_name = os.path.basename(file_path)
    nominal_data_list = []

    try:
        with open(file_path, 'r', encoding='utf-8') as data_file:
            for sentence in parse_incr(data_file):
                tokens_list_dicts = []
                # Process tokens within the sentence, handling IDs and filtering stopwords
                for token_data in sentence:
                    # Skip tokens without 'id' or with invalid IDs (e.g., multiword tokens)
                    if "id" not in token_data or token_data["id"] is None:
                         continue

                    token_id = convert_tuple_id_to_float(token_data["id"])
                    if token_id is None: # Skip if ID conversion failed
                        continue

                    # Standardize dependency relation before filtering stopwords
                    corrected_deprel = standardize_deprel(token_data.get("deprel")) # Use .get for safety

                    # Filter out stopwords based on standardized deprel (Section 5.3.1)
                    if corrected_deprel in stopwords_deprels:
                        continue

                    # Convert head ID to float as well
                    head_id = convert_tuple_id_to_float(token_data.get("head")) if token_data.get("head") is not None else None

                    # Store token data as a dictionary
                    token_representation = {
                         "id": token_id,
                         "form": token_data.get("form"),
                         "lemma": token_data.get("lemma"),
                         "upos": token_data.get("upos"),
                         "xpos": token_data.get("xpos"),
                         "feats": token_data.get("feats", {}), # Default to empty dict for morphological features
                         "head": head_id,
                         "deprel": corrected_deprel, # Use corrected deprel
                         "deps": token_data.get("deps"),
                         "misc": token_data.get("misc", {}), # Default to empty dict
                    }
                    tokens_list_dicts.append(token_representation)

                # Process nominal heads within the cleaned token list for this sentence
                nominal_data_list.extend(process_nominal_heads(tokens_list_dicts, file_name))

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    return nominal_data_list

# --- Main Processing Flow ---

if __name__ == "__main__":
    # List to hold combined nominal data from all files
    all_nominal_rows = []

    # Check if the data folder exists
    if not os.path.isdir(CONLLU_FOLDER):
        print(f"Error: CoNLL-U folder path not found: {CONLLU_FOLDER}")
        print("Please update CONLLU_FOLDER variable in the script configuration.")
    else:
        # Process all files in the specified folder
        print(f"Starting data processing from folder: {CONLLU_FOLDER}")
        # Use sorted listdir for consistent processing order
        for filename in sorted(os.listdir(CONLLU_FOLDER)):
            if filename.endswith(".conllu"):
                file_path = os.path.join(CONLLU_FOLDER, filename)
                nominal_data_from_file = process_conllu_file(file_path)
                all_nominal_rows.extend(nominal_data_from_file)

        # Create DataFrame from collected data
        nominals_df = pd.DataFrame(all_nominal_rows)
        print(f"\nFinished processing all files. Initial DataFrame shape: {nominals_df.shape}")

        # Drop duplicates based on all columns (Section 5.3.4)
        if not nominals_df.empty:
            original_rows = len(nominals_df)
            nominals_df.drop_duplicates(inplace=True)
            print(f"Dropped {original_rows - len(nominals_df)} duplicate nominal phrase entries. Shape after dropping duplicates: {nominals_df.shape}")

            # Adjust numerical values to prevent zeros if necessary (Section 5.3.2)
            nominals_df = adjust_numerical_values(nominals_df)

            # Save the cleaned nominal features to CSV (Section 6.4 mentions saving to CSV)
            try:
                nominals_df.to_csv(OUTPUT_CSV_PATH, index=False)
                print(f"Nominal features saved to {OUTPUT_CSV_PATH}")
            except Exception as e:
                print(f"Error saving the processed CSV file: {e}")
        else:
            print("No nominal phrase data extracted or all data was duplicated.")

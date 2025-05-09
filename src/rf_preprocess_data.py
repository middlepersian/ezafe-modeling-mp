# -*- coding: utf-8 -*-
# src/rf_preprocess_data.py
# Author: Raha Musavi
# Matrikelnummer: 08022255354
# Thesis: Modeling the Presence and the Classification of Ezafe in Middle Persian Nominal Phrases
# Date: 01 May 2025

"""
This script processes CoNLL-U files to extract linguistic features from nominal
head-dependent pairs in Middle Persian, aiming to create a dataset for machine
learning classification of ezafe constructions.

The processing pipeline involves the following stages:
1.  Defining utility functions for standardizing dependency labels and converting
    CoNLL-U token IDs (integers or tuples) to a consistent float format for
    numerical operations.
2.  Implementing a custom Token class to manage token attributes and include a
    method for resolving 'conj' dependency relations to identify the primary head.
3.  A preliminary pass over all specified CoNLL-U files is conducted to compute
    corpus-wide lemma frequency counts for potential nominal heads (nouns,
    pronouns, proper nouns) and dependents. This ensures global frequency
    data is available before feature extraction.
4.  A second, detailed pass processes each sentence to identify and extract
    features for valid nominal head-dependent pairs.
5.  Dependents are filtered to include only those with predefined non-clausal
    dependency relations and to exclude specific functional lemmas ('ī', 'ud').
6.  For each identified head-dependent pair, a comprehensive set of features
    is calculated, categorized as:
    - Identifiers (IDs, forms)
    - Morphosyntactic Properties (UPOS, deprel, number, NP dependency pattern)
    - Complexity Indicators (NP depth, counts of modifiers before/after head,
      number of dependents on head and dependent, linear distance)
    - Lexical and Frequency Features (head and dependent lemma frequency,
      modifier-to-head ratio)
    - Placement Features (relative position of head in sentence, position of
      dependent relative to head)
    - Text Property (source file)
7.  A binary 'ezafe_label' is assigned (1 if the dependent is governed by an
    ezafe marker 'ī', 0 otherwise).
8.  Extracted features for all nominal pairs are accumulated in a list of dictionaries.
9.  The collected nominal phrase features are saved to a CSV file
    ('rf_inputs.csv').
10. All token data processed during the second pass is also saved to a separate
    CSV file ('all_tokens.csv').

This script directly implements the preprocessing methodology detailed in Chapter 6
of the thesis, focusing on non-clausal dependency relationships and generating
features that capture syntactic structure, complexity, lexical characteristics,
and positional information relevant to ezafe analysis.
"""

import os
import pandas as pd
from collections import Counter
from conllu import parse_incr

# Define a dictionary for standardizing dependency labels to a consistent set.
standardized_deprels = {
    "nm": "nmod",
    "nmod:poss": "nmod",
    "nmod:det": "nmod",
    "adjmod": "amod",
    "al:relcl": "acl:relcl", # Standardize alternative forms of acl:relcl
    "acl:recl": "acl:relcl"
}

# Define dependency relations to be excluded from head-dependent pair analysis.
# These typically represent non-syntactic or structural markers.
stopwords_deprels = ['punct', 'reparandum']

def standardize_deprel(deprel):
    """
    Standardizes dependency relation strings based on a predefined mapping.

    Args:
        deprel (str): The original dependency relation string.

    Returns:
        str: The standardized dependency relation string, or the original if
             no specific rule applies.
    """
    return standardized_deprels.get(deprel, deprel)

def convert_tuple_id_to_float(id_tuple):
    """
    Converts a token ID, potentially in integer, string, or tuple format
    (as per CoNLL-U specifications for multi-word tokens or empty nodes),
    to a consistent float representation.

    This conversion supports consistent numerical processing and distance
    calculations across different ID types. Handles integer IDs and tuple
    IDs formatted as (integer, '.', decimal_part).

    Args:
        id_tuple (Union[int, Tuple[int, int, int], str]): The token ID from CoNLL-U data.

    Returns:
        Optional[float]: The converted float ID, or None if the input format
                         cannot be converted.
    """
    if isinstance(id_tuple, tuple) and len(id_tuple) == 3:
        # Handle tuple IDs, assuming a format like (integer, '.', decimal_part)
        integer_part = id_tuple[0]
        decimal_part = id_tuple[2]
        try:
            id_num = f"{int(integer_part)}.{int(decimal_part)}"
            return float(id_num)
        except (ValueError, TypeError):
            # Return None if tuple components cannot be converted to integers
            return None

    # Handle integer or float IDs
    try:
        return float(id_tuple)
    except (ValueError, TypeError):
         # Return None if direct float conversion fails
         return None


class Token:
    """
    Represents a single token from a CoNLL-U parsed sentence, storing
    attributes for standard CoNLL-U fields and resolved head/dependency data.
    """
    def __init__(self, id_, form, lemma, upos, xpos, feats, head, deprel, deps, misc):
        """
        Initializes a Token object with provided linguistic data.

        Args:
            id_ (Union[int, Tuple, str]): Token ID.
            form (str): Word form.
            lemma (str): Lemma.
            upos (str): Universal part-of-speech tag.
            xpos (str): Language-specific part-of-speech tag.
            feats (Dict): Morphological features dictionary.
            head (Union[int, Tuple, str]): Head of the current token.
            deprel (str): Universal dependency relation to the head.
            deps (List): List of secondary dependencies.
            misc (Dict): Miscellaneous features dictionary.
        """
        self.id = id_
        self.form = form
        self.lemma = lemma
        self.upos = upos
        self.xpos = xpos
        self.feats = feats if feats is not None else {}
        # Store original head and deprel, convert head to float upon initialization if possible
        self._original_head = head
        self._original_deprel = deprel
        self.head = convert_tuple_id_to_float(head)
        self.deprel = standardize_deprel(deprel) # Standardize deprel upon initialization

        self.deps = deps
        self.misc = misc
        # Initialize resolved attributes which will be updated after resolving conjuncts
        self._resolved_head = self.head
        self._resolved_deprel = self.deprel


    def resolve_conj(self, token_list):
        """
        Resolves the dependency relation for tokens marked as 'conj'.

        This method traverses the dependency tree upwards from a 'conj' token
        to find the first ancestor with a non-'conj' dependency relation. The
        head and dependency relation of this ancestor are then assigned as the
        resolved head and deprel for the current token. This clarifies the
        syntactic role of coordinated elements. Includes a cycle detection
        mechanism to prevent infinite loops during traversal.

        Args:
            token_list (List[Token]): A list containing all Token objects for the sentence.
        """
        # If the token's resolved deprel is not 'conj', no further resolution is needed
        if self._resolved_deprel != 'conj':
             # Re-assigning resolved head/deprel (redundant if not 'conj', but safe)
             self.head = self._resolved_head
             self.deprel = self._resolved_deprel
             return

        current_token = self
        # Track visited token IDs to detect cycles in the dependency graph
        path = []
        current_token_float_id = convert_tuple_id_to_float(current_token.id)
        if current_token_float_id is not None:
             path.append(current_token_float_id)

        # Traverse up the head chain. Continue as long as the current token's
        # original deprel is 'conj' and it is not the root (head != 0).
        while current_token is not None and convert_tuple_id_to_float(current_token._original_deprel) == 'conj' and convert_tuple_id_to_float(current_token.head) != 0:
            # Find the head token within the sentence's token list
            head_token = next((t for t in token_list if convert_tuple_id_to_float(t.id) == convert_tuple_id_to_float(current_token.head)), None)

            if head_token is None:
                # Stop if the designated head token is not found in the list
                break

            head_token_float_id = convert_tuple_id_to_float(head_token.id)
            # Check if the head ID is invalid or has already been visited in the path (indicating a cycle)
            if head_token_float_id is None or head_token_float_id in path:
                 if head_token_float_id is not None:
                     # Output a warning if a potential cycle is detected
                     print(f"Warning: Cycle or invalid head detected involving token ID {self.id} during conj resolution.")
                 break # Stop traversal due to cycle or invalid ID

            path.append(head_token_float_id) # Add the head's ID to the path
            current_token = head_token # Move to the head token and continue traversal

        # After the loop terminates, current_token is the token whose dependency
        # relation is not 'conj', or it is the root, or None if traversal stopped
        # unexpectedly (e.g., broken chain).
        # Determine the resolved head and dependency relation from the final token in the traversal path.
        resolved_head_float = convert_tuple_id_to_float(current_token.head) if current_token else self.head
        resolved_deprel_std = standardize_deprel(current_token.deprel) if current_token else self.deprel

        # Store the determined resolved head and dependency relation
        self._resolved_head = resolved_head_float
        self._resolved_deprel = resolved_deprel_std

        # Update the token's primary head and deprel attributes to the resolved values
        self.head = self._resolved_head
        self.deprel = self._resolved_deprel


class Sentence:
    """
    Represents a sentence container, holding a list of Token objects and associated metadata.
    """
    def __init__(self, sentence_id, tokens, metadata=None):
        """
        Initializes a Sentence object.

        Args:
            sentence_id (str): A unique identifier for the sentence.
            tokens (List[Token]): A list of Token objects belonging to this sentence.
            metadata (Dict, optional): Dictionary containing sentence-level metadata. Defaults to None.
        """
        self.sentence_id = sentence_id
        self.tokens = tokens
        self.metadata = metadata if metadata else {}

    def get_tokens(self):
        """
        Retrieves the list of Token objects within the sentence.
        """
        return self.tokens

# Global counters for lemma frequencies across the entire corpus.
# These are populated in the first pass and used to calculate features
# head_frequency and dependent_frequency in the second pass.
head_lemma_counts = Counter()
dependent_lemma_counts = Counter()

def update_lemma_counts(tokens):
    """
    Updates the global lemma frequency counters based on the Universal
    Part-of-Speech (UPOS) tag of each token.

    Tokens tagged as NOUN, PROPN, or PRON are counted as potential heads;
    all other UPOS tags contribute to the dependent counts.

    Args:
        tokens (List[Token]): A list of Token objects from a sentence.
    """
    for token in tokens:
        # Count lemmas based on their UPOS category
        if token.upos in {'NOUN', 'PROPN', 'PRON'}:
            head_lemma_counts[token.lemma] += 1
        else: # Tokens with other UPOS tags contribute to dependent counts
            dependent_lemma_counts[token.lemma] += 1

def get_np_depth(token, token_list, depth=0, max_depth=20, visited=None):
    """
    Recursively computes the syntactic depth of a nominal phrase structure
    originating from a given head token within a sentence's dependency tree.

    Depth is defined as the length of the longest head-dependent path starting
    from the token. Includes a mechanism to track visited nodes and limit
    recursion depth to handle complex or potentially problematic structures.

    Args:
        token (Token): The starting token, typically a nominal head.
        token_list (List[Token]): All Token objects in the current sentence.
        depth (int, optional): The current depth level in the recursion (initial call should use 0). Defaults to 0.
        max_depth (int, optional): The maximum allowed recursion depth. Defaults to 20.
        visited (Set[float], optional): A set of float token IDs visited in the current path to detect cycles. Defaults to None.

    Returns:
        int: The maximum syntactic depth observed within the phrase structure
             descending from the starting token.
    """
    if visited is None:
        visited = set()

    token_float_id = convert_tuple_id_to_float(token.id)
    # Base cases for recursion: invalid token ID, token already visited (cycle), or maximum depth reached
    if token_float_id is None or token_float_id in visited or depth >= max_depth:
        return depth

    visited.add(token_float_id) # Mark the current token as visited

    # Identify immediate dependents of the current token
    dependents = [t for t in token_list if convert_tuple_id_to_float(t.head) == token_float_id]

    # Base case: no dependents found for the current token
    if not dependents:
        return depth

    # Recursive step: Calculate depths for all valid dependents and find the maximum
    dependent_depths = [get_np_depth(dep, token_list, depth + 1, max_depth, visited)
                        for dep in dependents if convert_tuple_id_to_float(dep.id) is not None]

    # If no valid dependents were found after filtering, return the current depth
    if not dependent_depths:
         return depth
    else:
         # Return the maximum depth reached among all recursive calls, including the current level's base depth
         return max(dependent_depths + [depth])


def get_plurality(token_feats):
    """
    Extracts the 'Number' morphological feature (Singular or Plural) from a
    token's feature dictionary.

    Args:
        token_feats (Dict[str, Any]): The features dictionary of a token.

    Returns:
        str: The value of the "Number" feature ("Sing", "Plur"), defaulting to
             "Sing" if the feature is absent.
    """
    number = token_feats.get("Number", "Sing")
    return number


def process_nominal_heads(token_list, source_file_name):
    """
    Identifies potential nominal heads within a sentence and extracts features
    for their valid head-dependent pairs.

    Potential nominal heads are tokens with UPOS NOUN, PROPN, or PRON that are
    not sentence roots. Valid dependents are filtered based on a predefined list
    of non-clausal dependency relations and excluded lemmas ('ī', 'ud').
    Features are computed for each resulting head-dependent pair.

    Args:
        token_list (List[Token]): A list of Token objects representing a sentence.
        source_file_name (str): The name of the source CoNLL-U file from which
                                the sentence was extracted.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                              contains the extracted features for one nominal
                              head-dependent pair.
    """
    nominal_data = [] # List to store feature dictionaries for nominal pairs
    potential_ezafe_markers = {'ī'} # Set of lemmas identified as ezafe markers

    # Define the set of valid dependency relations for dependents to be included
    # in nominal head-dependent pair analysis. These represent core nominal modifiers.
    valid_dependent_deprels = ['nmod', 'amod', 'flat', 'cc', 'det', 'clf', 'nummod', 'advmod', 'appos']

    # Identify potential nominal heads within the sentence (NOUN, PROPN, PRON) that are not sentence roots.
    potential_nominal_heads = [t for t in token_list if t.upos in {'NOUN', 'PROPN', 'PRON'} and convert_tuple_id_to_float(t.head) != 0]

    # Process each potential nominal head
    for nominal_head in potential_nominal_heads:
        nominal_head_id_float = convert_tuple_id_to_float(nominal_head.id)
        # Skip the head if its ID cannot be converted to a float
        if nominal_head_id_float is None:
            continue

        # Find all immediate dependents of the current nominal head
        all_head_dependents = [t for t in token_list if convert_tuple_id_to_float(t.head) == nominal_head_id_float]

        # Filter dependents to include only those with dependency relations from
        # the valid list and exclude specific lemmas ('ī', 'ud').
        dependents_for_pairs = [t for t in all_head_dependents if t.deprel in valid_dependent_deprels and t.lemma not in ['ī', 'ud']]

        # If the nominal head has no eligible dependents after filtering, skip this head
        if not dependents_for_pairs:
            continue

        # --- Calculate Head-specific and Sentence-level Features ---
        # These features are calculated once for the current nominal head within its sentence context.

        # Count the number of nouns/proper nouns and adjectives linearly preceding the head.
        num_nouns_before = sum(1 for t in token_list if t.upos in {'NOUN', 'PROPN'} and convert_tuple_id_to_float(t.id) is not None and convert_tuple_id_to_float(t.id) < nominal_head_id_float)
        num_adjs_before = sum(1 for t in token_list if t.upos == 'ADJ' and convert_tuple_id_to_float(t.id) is not None and convert_tuple_id_to_float(t.id) < nominal_head_id_float)
        # Count the number of adjectives linearly following the head.
        num_adjs_after = sum(1 for t in token_list if t.upos == 'ADJ' and convert_tuple_id_to_float(t.id) is not None and convert_tuple_id_to_float(t.id) > nominal_head_id_float)

        # Determine the syntactic depth of the phrase structure rooted at this nominal head.
        np_depth = get_np_depth(nominal_head, token_list)

        # Extract the number marking feature from the head's morphological features.
        head_number = get_plurality(nominal_head.feats)

        # Count the total number of immediate dependents attached to the head.
        num_dependents_head = len(all_head_dependents)

        # Construct a string representing the sequence of dependency relations
        # of all immediate dependents of the head.
        np_deprel_pattern = " ".join(t.deprel for t in all_head_dependents)

        # Calculate the relative linear position of the head within the sentence, normalized between 0 and 1.
        # A small value (1e-5) is added to the denominator to prevent division by zero for empty sentences.
        relative_position = nominal_head_id_float / (len(token_list) if len(token_list)>0 else 1e-5)


        # --- Iterate through eligible Dependents ---
        # For each valid dependent, create a head-dependent pair and calculate pair-specific features.
        for dependent in dependents_for_pairs:
            dependent_id_float = convert_tuple_id_to_float(dependent.id)
            # Skip the dependent if its ID cannot be converted to a float
            if dependent_id_float is None:
                continue

            # Calculate the absolute linear token distance between the head and the dependent.
            distance = abs(dependent_id_float - nominal_head_id_float)

            # Determine the linear positional order of the dependent relative to the head.
            # 1 indicates the dependent precedes the head; 2 indicates the dependent follows the head.
            position = 1 if nominal_head_id_float < dependent_id_float else 2

            # Check if the dependent is itself headed by an ezafe marker ('ī'),
            # indicating that the current dependent is an ezafe construction.
            has_ezafe = any(t.lemma in potential_ezafe_markers and convert_tuple_id_to_float(t.head) == dependent_id_float for t in token_list)

            # Retrieve the corpus-wide frequency counts for the head's lemma and the dependent's lemma.
            head_freq = head_lemma_counts.get(nominal_head.lemma, 0)
            dependent_freq = dependent_lemma_counts.get(dependent.lemma, 0) # Using dependent_freq for clarity

            # Calculate the ratio of the dependent's lemma frequency to the head's lemma frequency.
            # A small value (1e-5) is added to the head frequency to prevent division by zero.
            modifier_to_head_ratio = dependent_freq / (head_freq + 1e-5)

            # Count the number of immediate dependents attached to the current dependent token.
            num_dependents_dependent = len([t for t in token_list if convert_tuple_id_to_float(t.head) == dependent_id_float])

            # Append the dictionary of extracted features for this head-dependent pair to the list.
            nominal_data.append({
                'nominal_head_id': nominal_head_id_float,
                'nominal_head_form': nominal_head.form,
                'nominal_head_upos': nominal_head.upos,
                'nominal_head_deprel': nominal_head.deprel, # Using the resolved deprel
                'head_number': head_number,
                'head_frequency': head_freq,
                'dependent_frequency': dependent_freq, # Using dependent_frequency name
                'modifier_to_head_ratio': modifier_to_head_ratio,
                'dependent_id': dependent_id_float,
                'dependent_form': dependent.form,
                'dependent_upos': dependent.upos, # Using the token's UPOS
                'dependent_deprel': dependent.deprel, # Using the resolved deprel
                'distance': distance,
                'position': position, # 1: dependent precedes head; 2: dependent follows head (Consistent with thesis Fig 27)
                'num_nouns_before': num_nouns_before,
                'num_adjs_before': num_adjs_before,
                'num_adjs_after': num_adjs_after,
                'np_depth': np_depth,
                'num_dependents_head': num_dependents_head,
                'num_dependents_dependent': num_dependents_dependent,
                'ezafe_label': int(has_ezafe), # Binary label (0 or 1)
                'relative_position_in_sent': relative_position,
                'np_deprel_pattern': np_deprel_pattern,
                'source_file': source_file_name
            })

    return nominal_data


def process_conllu_file(file_path):
    """
    Processes a single CoNLL-U file to parse sentences, build Token objects,
    resolve dependency relations (specifically 'conj'), extract features for
    nominal head-dependent pairs, and collect data for all tokens.

    Args:
        file_path (str): The full path to the CoNLL-U file to be processed.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: A tuple containing:
            - A list of dictionaries, each representing the features for a
              nominal head-dependent pair.
            - A list of dictionaries, each containing data for a single token
              from the file.
        Returns empty lists if an error occurs during file processing.
    """
    file_name = os.path.basename(file_path)
    nominal_rows = [] # List to store nominal phrase feature dictionaries
    token_rows_per_file = [] # List to store dictionaries for all tokens

    try:
        # Open and parse the CoNLL-U file sentence by sentence using parse_incr
        with open(file_path, 'r', encoding='utf-8') as data_file:
            for sentence in parse_incr(data_file):
                tokens_list_orig = [] # Temporary list to hold custom Token objects for the current sentence
                # Generate a sentence ID from metadata or create a unique default ID
                sentence_idx = sentence.metadata.get("sent_id", f"sent_{len(nominal_rows)}_{len(token_rows_per_file)}_file_{file_name}")

                # Extract sentence-level metadata
                sentence_text = sentence.metadata.get("text", "")
                newpart = sentence.metadata.get("newpart", "")
                translation = sentence.metadata.get("translation", "")

                # Iterate through raw token data to create custom Token objects
                for token_data in sentence:
                     # Attempt to convert the token ID to float
                     token_id_float = convert_tuple_id_to_float(token_data['id'])
                     # Only create a Token object if the ID is valid and convertible
                     if token_id_float is not None:
                        # Attempt to convert head ID to float as well
                        head_id_float = convert_tuple_id_to_float(token_data["head"])

                        # Create a Token object using original CoNLL-U data
                        token = Token(
                            id_=token_data['id'], # Preserve original ID format
                            form=token_data["form"],
                            lemma=token_data["lemma"],
                            upos=token_data["upos"],
                            xpos=token_data["xpos"],
                            feats=token_data.get("feats", {}),
                            head=token_data["head"], # Preserve original head format for Token init
                            deprel=token_data["deprel"], # Preserve original deprel format for Token init
                            deps=token_data["deps"],
                            misc=token_data.get("misc", {}),
                        )
                        tokens_list_orig.append(token)

                        # Store basic token data. Resolved head/deprel will be added after resolve_conj.
                        token_rows_per_file.append({
                            "id": token_id_float, # Use float ID for output consistency
                            "original_id": token.id, # Include original ID format
                            "form": token.form,
                            "lemma": token.lemma,
                            "upos": token.upos,
                            "xpos": token.xpos,
                            "feats": token.feats,
                            "head": token.head, # Float head from Token init
                            "deprel": token.deprel, # Standardized deprel from Token init
                            "misc": token.misc,
                            "sent_id": sentence_idx,
                            "text": sentence_text,
                            "newpart": newpart,
                            "translation": translation,
                            "source_file": file_name,
                        })

                # After creating all Token objects for the sentence, resolve 'conj' dependencies.
                # This updates the 'head' and 'deprel' attributes of relevant Token objects.
                for token in tokens_list_orig:
                     token.resolve_conj(tokens_list_orig)

                # Update the token data dictionaries with the resolved head and dependency relations.
                token_id_map = {convert_tuple_id_to_float(t.id): t for t in tokens_list_orig if convert_tuple_id_to_float(t.id) is not None}
                for row in token_rows_per_file:
                    token_float_id = row.get('id') # Use .get() for safety
                    if token_float_id in token_id_map:
                        resolved_token = token_id_map[token_float_id]
                        row['resolved_head'] = resolved_token.head
                        row['resolved_deprel'] = resolved_token.deprel


                # Filter the list of Token objects for nominal phrase feature extraction.
                # Exclude tokens with dependency relations in the stopwords list and those with invalid heads.
                filtered_tokens_for_np_processing = [
                    t for t in tokens_list_orig
                    if t.deprel not in stopwords_deprels # Filter using the resolved deprel
                    and convert_tuple_id_to_float(t.head) is not None # Ensure the resolved head ID is valid
                    and convert_tuple_id_to_float(t.id) is not None # Ensure the token itself has a valid ID
                ]

                # Extract nominal phrase features from the filtered tokens and extend the global list.
                nominal_rows.extend(process_nominal_heads(filtered_tokens_for_np_processing, file_name))

    except Exception as e:
        # Print an error message if processing the file fails and return empty lists.
        print(f"Error processing file {file_path}: {e}")
        return [], []

    # Return the collected nominal features and token data for the file.
    return nominal_rows, token_rows_per_file


# Define the path to the directory containing the CoNLL-U files.
# >>> IMPORTANT: Update this path to point to the actual directory on your system. <<<
folder_path = cd path/to/your/project # <--- *** UPDATE THIS PATH ***

# Initialize lists to collect data from all processed files.
all_nominal_rows = []
all_token_rows = []
# Re-initialize global lemma frequency counters before the first pass.
head_lemma_counts = Counter()
dependent_lemma_counts = Counter()

# --- First Pass: Collect tokens and build corpus-wide lemma frequency counts ---
# This initial pass reads through all files solely to gather lemma counts.
# These global counts are required for calculating frequency-based features
# accurately in the second pass.
print("--- First pass: Collecting all tokens and lemma counts ---")
if os.path.exists(folder_path):
    # Get a list of all files ending with '.conllu' in the specified directory.
    conllu_files = [f for f in os.listdir(folder_path) if f.endswith(".conllu")]
    # Check if any .conllu files were found.
    if not conllu_files:
        print("No .conllu files found in the directory.")
else:
    print(f"Directory {folder_path} does not exist.")
    conllu_files = [] # Ensure the file list is empty if the directory doesn't exist.

# Iterate through each CoNLL-U file during the first pass.
for filename in conllu_files:
    file_path = os.path.join(folder_path, filename)
    print(f"Collecting tokens and counts from {filename}...")
    try:
        # Open and parse the file sentence by sentence.
        with open(file_path, 'r', encoding='utf-8') as data_file:
             for sentence in parse_incr(data_file):
                  sentence_tokens = [] # Temporary list for tokens in the current sentence for counting.
                  # Process each token in the sentence.
                  for token_data in sentence:
                       # Attempt to convert token ID to float.
                       token_id_float = convert_tuple_id_to_float(token_data['id'])
                       if token_id_float is not None:
                            # Standardize dependency relation.
                            standardized_deprel = standardize_deprel(token_data["deprel"])
                            # Exclude tokens with specified stopwords dependency relations from counting.
                            if standardized_deprel not in stopwords_deprels:
                                 # Attempt to convert head ID to float.
                                 head_id_float = convert_tuple_id_to_float(token_data["head"])
                                 # Create a basic Token object for frequency counting purposes.
                                 token = Token(token_id_float, token_data["form"], token_data["lemma"], token_data["upos"], token_data["xpos"], token_data.get("feats", {}), head_id_float, standardized_deprel, token_data["deps"], token_data.get("misc", {}))
                                 sentence_tokens.append(token) # Add token to temporary list.

                  # Update the global lemma counts using tokens from the current sentence.
                  update_lemma_counts(sentence_tokens)

    except Exception as e:
         # Print an error message if token collection fails for a file.
         print(f"Error collecting tokens from {file_path}: {e}")

# --- Second Pass: Extract detailed nominal phrase features ---
# This pass iterates through the files again to perform detailed feature extraction
# for nominal head-dependent pairs, using the lemma counts gathered previously.
print("\n--- Second pass: Extracting nominal phrase features ---")
# Proceed only if the directory exists and files were found in the first pass.
if os.path.exists(folder_path) and conllu_files:
    for filename in conllu_files:
        file_path = os.path.join(folder_path, filename)
        print(f"Processing {filename} for features...")
        # Call the processing function for the file to get nominal features and token data.
        nominal_rows_file, token_rows_file = process_conllu_file(file_path)
        # Extend the global lists with the results from the current file.
        all_nominal_rows.extend(nominal_rows_file)
        all_token_rows.extend(token_rows_file)

else:
     # Print informative messages if the second pass is skipped.
     if not os.path.exists(folder_path):
          print(f"Directory {folder_path} does not exist. Second pass skipped.")
     elif not conllu_files:
          print("No .conllu files were found during the first pass. Second pass skipped.")


# --- Save Collected DataFrames to CSV Files ---
# Convert the collected lists of dictionaries into pandas DataFrames and save them.

# Save the extracted nominal phrase features to a CSV file.
if all_nominal_rows:
    nominals_df = pd.DataFrame(all_nominal_rows)
    output_features_file = "rf_inputs.csv" # Renamed output file as requested in thesis text (Table 19 comparison implies a file for the RF model inputs)
    nominals_df.to_csv(output_features_file, index=False)
    print(f"\nNominal phrase features saved to {output_features_file}")

    # Display the data types of the DataFrame columns for verification.
    print("\nDataFrame dtypes:")
    print(nominals_df.dtypes)
else:
    print("\nNo nominal data was extracted during processing. The features CSV file was not created.")

# Save all collected token data to a separate CSV file.
if all_token_rows:
    token_df = pd.DataFrame(all_token_rows)
    output_tokens_file = "all_tokens.csv"
    token_df.to_csv(output_tokens_file, index=False)
    print(f"All token data saved to {output_tokens_file}")
else:
    print("No token data was extracted during processing. The tokens CSV file was not created.")
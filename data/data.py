"""
This create random dataset to train our models.
"""
# -------------------------------------------------------------------------------------------------------------------- #
# Standard Library
from typing import List, Union

# Third Party
import numpy as np
import pandas as pd

# Private

# -------------------------------------------------------------------------------------------------------------------- #

# Code


def create_general_data():
    # Set the dimensions of the dataset
    num_cell_lines = 1000
    num_genes = 100
    num_drugs = 10
    # Generate random transcriptomic data (gene expression values)
    transcriptomic_data = np.random.rand(num_cell_lines, num_genes)
    # Generate random drug response scores
    drug_response_scores = np.random.rand(num_cell_lines, num_drugs)
    # Generate random target labels (drug classes)
    target_labels = np.random.randint(1, num_drugs + 1, size=num_cell_lines)
    # Create column names for genes, drugs, and target label
    gene_columns = [f"Gene{i + 1}" for i in range(num_genes)]
    drug_columns = [f"Drug{i + 1}" for i in range(num_drugs)]
    target_column = "Target"
    # Combine the data into a single dataset
    data = np.concatenate(
        (transcriptomic_data, drug_response_scores, target_labels.reshape(-1, 1)),
        axis=1,
    )
    # Create a pandas DataFrame
    df = pd.DataFrame(data, columns=gene_columns + drug_columns + [target_column])
    return df

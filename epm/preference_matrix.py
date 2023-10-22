from typing import List
from dataclasses import dataclass

import numpy as np

@dataclass
class PM:
    pm: np.ndarray

    def __hash__(self):
        return hash(self.pm.tostring())

def ranking_to_preference_matrix(ranking, labels):
    """
    Convert a ranking to a preference matrix.

    Parameters:
        ranking (str) - Ranking in the format 'a>b>c>d'
        labels (List[str]) - List of labels

    Returns:
        matrix (np.ndarray) - Preference Matrix
    """
    # Create matrix
    num_labels = len(labels)
    matrix = np.zeros((num_labels, num_labels))

    # Fill matrix
    elements = ranking.split('>')
    for row, label1 in enumerate(labels):
        for col, label2 in enumerate(labels):
            if col <= row:
                continue  # Skip lower triangular part and diagonal
            
            rank1 = next((index for index, item in enumerate(elements) if label1 in item), None)
            rank2 = next((index for index, item in enumerate(elements) if label2 in item), None)
            
            if rank1 is None or rank2 is None:
                matrix[row, col] = np.nan
            elif rank1 < rank2:
                matrix[row, col] = 1
                matrix[col, row] = -1
            elif rank1 > rank2:
                matrix[row, col] = -1
                matrix[col, row] = 1

    return PM(matrix)

def aggregate_preference_matrix(preference_matrices: List[PM], technique: str):
    """
    Calculate the (nan)mean or mode preference matrix of a list of preference matrices.

    Parameters:
        preference_matrices (List[np.ndarray]) - List of preference matrices
        technique (str) - Technique to aggregate preference matrices

    Returns:
        mean_preference_matrix (np.ndarray) - Mean preference matrix
    """
    pms = [pm.pm for pm in preference_matrices]

    if technique == 'mean':
        return PM(np.nanmean(pms, axis=0))
    elif technique == 'mode':
        return matrix_mode(pms)
    else:
        raise ValueError(f"Invalid aggregate technique: `{technique}`")

def matrix_mode(preference_matrices: List[np.ndarray]):
    """
    Calculate the mode preference matrix of a list of preference matrices.

    Parameters:
        preference_matrices (List[np.ndarray]) - List of preference matrices

    Returns:
        mode_preference_matrix (np.ndarray) - Mode preference matrix
    """
    # Stack the matrices along a new axis to create a 3D array
    stacked_matrices = np.stack(preference_matrices)

    # Count occurrences of -1, 0, and 1 in the stacked matrices
    counts = np.stack([
        np.sum(stacked_matrices == -1, axis=0),
        np.sum(stacked_matrices == 0, axis=0),
        np.sum(stacked_matrices == 1, axis=0)
    ], axis=0)

    # Determine the mode along the third axis
    mode_indices = np.argmax(counts, axis=0)
    
    # Map mode indices to corresponding values (-1, 0, 1)
    mode_preference_matrix = np.choose(mode_indices, [-1, 0, 1])

    return PM(mode_preference_matrix)

def distance_matrix(matrix_d: PM, matrix_s: PM):
    """
    Calculate the distance matrix between the dataset preference matrix
    and the subgroup preference matrix.

    Parameters:
        matrix_d (np.ndarray) - Dataset preference matrix
        matrix_s (np.ndarray) - Subgroup preference matrix

    Returns:
        matrix_l (np.ndarray) - Distance matrix
    """
    return .5 * (matrix_d.pm - matrix_s.pm)
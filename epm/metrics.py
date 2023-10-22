import numpy as np

from epm.preference_matrix import distance_matrix
from epm.subgroup import Subgroup

def rw_norm(dataset: Subgroup, item: Subgroup):
    """
    Compute the Rankingwise Norm quality measure.

    Parameters:
        matrix_d (np.ndarray) - Dataset preference matrix
        matrix_s (np.ndarray) - Subgroup preference matrix
        size_s (int) - Size of subgroup
        size_n (int) - Size of dataset

    Returns:
        rw_norm (float) - Rankingwise Norm
    """
    normalization, matrix_d, matrix_s = split(dataset, item)

    matrix_l = distance_matrix(matrix_d, matrix_s)
    
    return normalization * np.linalg.norm(matrix_l, ord='fro')

def rw_cov(dataset: Subgroup, item: Subgroup):
    """
    Compute the Rankingwise Covariance quality measure.

    Parameters:
        matrix_d (np.ndarray) - Dataset preference matrix
        matrix_s (np.ndarray) - Subgroup preference matrix
        size_s (int) - Size of subgroup
        size_n (int) - Size of dataset

    Returns:
        rw_cov (float) - Rankingwise Covariance
    """
    normalization, matrix_d, matrix_s = split(dataset, item)

    vector_d = matrix_d.pm.flatten()
    vector_s = matrix_s.pm.flatten()

    return -normalization * np.cov(vector_d, vector_s)[0, 1]

def lw_norm(dataset: Subgroup, item: Subgroup):
    """
    Compute the Labelwise Norm quality measure.

    Parameters:
        matrix_d (np.ndarray) - Dataset preference matrix
        matrix_s (np.ndarray) - Subgroup preference matrix
        size_s (int) - Size of subgroup
        size_n (int) - Size of dataset

    Returns:
        lw_norm (float) - Labelwise Norm
    """
    normalization, matrix_d, matrix_s = split(dataset, item)

    matrix_l = distance_matrix(matrix_d, matrix_s)

    max_val = 0

    for i in range(matrix_l.shape[0]):
        sum_i_j = 0
        for j in range(matrix_l.shape[1]):
            sum_i_j += matrix_l[i, j]**2
        max_val = max(max_val, sum_i_j)

    return normalization * max_val

def pw_max(dataset: Subgroup, item: Subgroup):
    """
    Compute the Pairwise Max quality measure.

    Parameters:
        matrix_d (np.ndarray) - Dataset preference matrix
        matrix_s (np.ndarray) - Subgroup preference matrix
        size_s (int) - Size of subgroup
        size_n (int) - Size of dataset

    Returns:
        pw_max (float) - Pairwise Max
    """
    normalization, matrix_d, matrix_s = split(dataset, item)

    matrix_l = distance_matrix(matrix_d, matrix_s)

    max_val = 0

    for i in range(matrix_l.shape[0]):
        for j in range(matrix_l.shape[1]):
            max_val = max(max_val, abs(matrix_l[i, j]))

    return normalization * max_val

def split(dataset: Subgroup, item: Subgroup):

    size_n = dataset.data.shape[0]
    size_s = item.data.shape[0]

    return np.sqrt(size_s/size_n), dataset.pm, item.pm

metrics = dict(
    rw_norm=rw_norm,
    rw_norm_mode=rw_norm,
    rw_cov=rw_cov,
    lw_norm=lw_norm,
    pw_max=pw_max
)
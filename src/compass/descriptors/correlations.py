# Created by gonzalezroy at 6/24/24
"""
Functions related to the calculation of correlations
"""
import numpy as np
from numba import njit


@njit(parallel=True, fastmath=True)
def calculate_mi_and_gc(cov_mat, num_atoms_per_residue):
    """
    Calculate Mutual Information (MI) scores and Generalized Correlation (GC) matrix.

    Args:
        cov_mat: covariance matrix
        num_atoms_per_residue: number of atoms per residue

    Returns:
        MI_scores: Mutual Information scores
        GC_matrix: Generalized Correlation matrix
    """
    num_residues = cov_mat.shape[0] // 3
    MI_scores = np.zeros((num_residues, num_residues), dtype=np.float32)
    GC_matrix = np.zeros((num_residues, num_residues), dtype=np.float32)

    for i in range(num_residues):
        for j in range(i, num_residues):
            cov_ij = np.float32(0)
            var_i = np.float32(0)
            var_j = np.float32(0)

            for k in range(num_atoms_per_residue):
                for l in range(num_atoms_per_residue):
                    idx_i = i * num_atoms_per_residue + k
                    idx_j = j * num_atoms_per_residue + l
                    cov_ij += cov_mat[idx_i, idx_j]
                    if k == l:
                        var_i += cov_mat[idx_i, idx_i]
                        var_j += cov_mat[idx_j, idx_j]
            cov_ij /= num_atoms_per_residue ** 2
            var_i /= num_atoms_per_residue
            var_j /= num_atoms_per_residue
            div_term = (cov_ij ** 2) / (var_i * var_j)
            MI_score = np.float32(0.5 * np.log(1 + div_term))
            MI_scores[i, j] = MI_scores[j, i] = MI_score
            exp_term = np.exp(-2 * MI_score)
            GC_matrix[i, j] = GC_matrix[j, i] = np.sqrt(1 - exp_term)

    return MI_scores, GC_matrix


def compute_gc_matrix(corr_coords, num_atoms_per_residue=1):
    """
    Compute the Generalized Correlation (GC) matrix.

    Args:
        corr_coords: Coordinates for computing correlations
        num_atoms_per_residue: Number of atoms per residue

    Returns:
        MI_scores: Mutual Information scores
        GC_matrix: Generalized Correlation matrix
    """
    # Compute covariance matrix from trajectory coordinates
    cov_matrix = compute_cov_matrix_trajectory(corr_coords)
    # Calculate Mutual Information (MI) scores and Generalized Correlation (GC) matrix
    test_matrix = cov_matrix[:2, :2]
    _, _ = calculate_mi_and_gc(test_matrix, num_atoms_per_residue)
    MI_scores, GC_matrix = calculate_mi_and_gc(cov_matrix,
                                               num_atoms_per_residue)
    # print("calculating generalised correlations here")
    return MI_scores, GC_matrix


def compute_cov_matrix_trajectory(coords):
    """
    Compute covariance matrix from trajectory coordinates.

    Args:
        coords: trajectory coordinates

    Returns:
        cov_matrix: covariance matrix
    """
    flat_coords = coords.reshape(coords.shape[0], -1)
    cov_matrix = np.cov(flat_coords.T)
    return cov_matrix

# Created by gonzalezroy at 6/24/24
"""
Functions related to the calculation of correlations
"""
import numpy as np
from numba import njit


@njit(parallel=False, fastmath=True)
def calculate_mi_and_gc(cov_mat, num_atoms_per_residue):
    """
    Fully vectorized version - computes all residue pairs at once.
    Best for moderate number of residues (<1000).
    TODO: Use chunks for residues > 1000
    """
    num_residues = cov_mat.shape[0] // num_atoms_per_residue
    n = num_atoms_per_residue  # Number of atoms per residue

    # Reshape covariance matrix into (num_residues, n, num_residues, n)
    cov_reshaped = cov_mat.reshape(num_residues, n, num_residues, n)

    # Sum over atom dimensions to get residue-level covariances
    # Shape: (num_residues, num_residues)
    cov_ij = np.sum(cov_reshaped, axis=(1, 3)) / (n ** 2)

    # Extract diagonal blocks for variances
    var_residues = np.zeros(num_residues, dtype=np.float32)
    for i in range(num_residues):
        i_start = i * n
        i_end = i_start + n
        var_residues[i] = np.sum(np.diag(cov_mat[i_start:i_end, i_start:i_end])) / n

    # Broadcast variances to create var_i and var_j matrices
    var_i = var_residues[:, np.newaxis]  # Shape: (num_residues, 1)
    var_j = var_residues[np.newaxis, :]  # Shape: (1, num_residues)

    # Compute MI and GC for all pairs
    div_term = (cov_ij ** 2) / (var_i * var_j + 1e-10)
    MI_scores = np.float32(0.5 * np.log(1.0 + div_term))

    exp_term = np.exp(-2.0 * MI_scores)
    GC_matrix = np.sqrt(1.0 - exp_term).astype(np.float32)

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

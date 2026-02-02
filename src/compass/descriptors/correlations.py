# Created by gonzalezroy at 6/24/24
"""
Functions related to the calculation of correlations
"""
import numpy as np
from numba import njit, prange


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
    num_residues = cov_mat.shape[0] // num_atoms_per_residue
    MI_scores = np.zeros((num_residues, num_residues), dtype=np.float32)
    GC_matrix = np.zeros((num_residues, num_residues), dtype=np.float32)

    # Precompute normalization factors
    norm_factor = np.float32(1.0 / (num_atoms_per_residue ** 2))
    var_norm_factor = np.float32(1.0 / num_atoms_per_residue)

    # Process upper triangle in parallel
    for i in prange(num_residues):
        # Define atom index range for residue i
        i_start = i * num_atoms_per_residue
        i_end = i_start + num_atoms_per_residue

        # Extract covariance block for residue i with itself (for var_i)
        cov_i_block = cov_mat[i_start:i_end, i_start:i_end]
        var_i = np.float32(np.sum(np.diag(cov_i_block)) * var_norm_factor)

        for j in range(i, num_residues):
            # Define atom index range for residue j
            j_start = j * num_atoms_per_residue
            j_end = j_start + num_atoms_per_residue

            # Extract covariance block between residues i and j
            cov_ij_block = cov_mat[i_start:i_end, j_start:j_end]
            cov_ij = np.float32(np.sum(cov_ij_block) * norm_factor)

            # Calculate var_j
            if i == j:
                var_j = var_i
            else:
                cov_j_block = cov_mat[j_start:j_end, j_start:j_end]
                var_j = np.float32(np.sum(np.diag(cov_j_block)) * var_norm_factor)

            # Calculate MI and GC
            div_term = (cov_ij ** 2) / (var_i * var_j + 1e-10)  # Small epsilon for numerical stability
            MI_score = np.float32(0.5 * np.log(1.0 + div_term))
            MI_scores[i, j] = MI_scores[j, i] = MI_score

            exp_term = np.exp(-2.0 * MI_score)
            GC_matrix[i, j] = GC_matrix[j, i] = np.sqrt(1.0 - exp_term)

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

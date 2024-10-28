import time
import numpy as np
from numba import njit, prange
from numpy import concatenate as concat
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorly as tl
from tensorly.decomposition import tucker
from compass.descriptors import geometry as geom


def reshape_matrices(matrices):
    """
    Reshape the matrices to be concatenated for Tucker decomposition

    Args:
        matrices: List of matrices to be reshaped

    Returns:
        reshaped_matrices: List of reshaped matrices
    """
    # Ensure all matrices have the same shape
    shapes = [matrix.shape for matrix in matrices]
    if len(set(shapes)) > 1:
        raise ValueError("All matrices must have the same shape")

    return matrices  # Matrices are reshaped after Tucker reconstruction


def perform_tucker(matrices, r=3):
    """
    Perform Tucker decomposition and reconstruction on the matrices

    Args:
        matrices: List of matrices to perform Tucker decomposition
        rank: Rank for Tucker decomposition

    Returns:
        tucker_reconstructed_avg: Average of Tucker-reconstructed tensors, scaled to [0, 1]
    """
    tucker_reconstructions = []
    ranks = []

    for matrix in matrices:
        core, factors_tucker = tucker(matrix, rank=[r, r, r])
        tucker_reconstructed = tl.tucker_to_tensor((core, factors_tucker))
        tucker_reconstructions.append(tucker_reconstructed)

    # Calculate the average of the Tucker-reconstructed tensors
    tucker_reconstructed_avg = np.mean(tucker_reconstructions, axis=0)

    # Scale the result to be between 0 and 1
    tensor_min = np.min(tucker_reconstructed_avg)
    tensor_max = np.max(tucker_reconstructed_avg)
    tucker_reconstructed_avg = (tucker_reconstructed_avg - tensor_min) / (tensor_max - tensor_min)

    return tucker_reconstructed_avg



def run_tucker(arg, matrices, n, first_timer):
    """
    Perform Tucker decomposition on the selected matrices and generate adjacency matrix

    Args:
        arg: namespace with the arguments
        matrices: list of matrices to be used in Tucker decomposition
        n: number of residues
        first_timer: initial time for time tracking
    """
    # Select the matrices to be used in the decomposition
    gc_mat = matrices["GC"]["data"]
    int_mat = matrices["INTERACTIONS"]["data"]
    cp_mat = matrices["COMMPROP"]["data"]
    dist_mat = matrices["MINDIST"]["data"]
    matrices = [gc_mat, int_mat, cp_mat]

    # Reshape and prepare matrices
    matrices = reshape_matrices(matrices)

    # Perform Tucker decomposition and compute the average
    tucker_result = perform_tucker(matrices)
    
    # Generate adjacency matrix based on the Tucker result
    adj_mat = tucker_result
    #adj_mat = adj_mat_raw.toarray()

    adj_name = geom.get_matrix_name(arg.out_dir, arg.title, "ADJACENCY")
    adj_mat = geom.save_matrix(adj_mat, n, adj_name, norm=True, prec=4)
    matrix_name = "Adjacency matrix"
    geom.plot_matrix(adj_mat, matrix_name, adj_name.replace(".mat", ".png"))

    tucker_time = round(time.time() - first_timer, 2)
    print(f"Until Tucker & Adjacency matrix computing: {tucker_time} s")
    return adj_name


# Created by gonzalezroy at 6/26/24

import numpy as np
from numba import njit, prange
from numpy import concatenate as concat
from scipy.sparse import lil_matrix
from sklearn.decomposition import PCA


def reshape_matrices(matrices):
    """
    Reshape the matrices to be concatenated for PCA

    Args:
        matrices: List of matrices to be reshaped

    Returns:
        data: Concatenated and reshaped matrices
    """
    # Ensure all matrices have the same shape
    shapes = [matrix.shape for matrix in matrices]
    if len(set(shapes)) > 1:
        raise ValueError("All matrices must have the same shape")

    # Concatenate the flattened matrices for PCA
    num_rows = shapes[0][0]
    flatened = [matrix.reshape(num_rows, -1) for matrix in matrices]
    data = concat(flatened, axis=1)
    return data


def perform_pca(data):
    """
    Perform PCA on the data

    Args:
        data: Data to perform PCA on

    Returns:
        pca_result: PCA result
    """
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    return pca_result


def calc_adjacency_matrix(pca_result, threshold=0.3, chunk_size=100):
    """
    Compute the adjacency matrix from the PCA results

    Args:
        pca_result: PCA result
        threshold: threshold for the adjacency matrix construction
        chunk_size: chunk size for parallel processing

    Returns:
        adjacency_matrix: Adjacency matrix
    """
    n_samples = pca_result.shape[0]
    adjacency_matrix = lil_matrix((n_samples, n_samples))

    for i in range(0, n_samples, chunk_size):
        for j in range(i, n_samples, chunk_size):
            end_i = min(i + chunk_size, n_samples)
            end_j = min(j + chunk_size, n_samples)
            chunk_i = pca_result[i:end_i]
            chunk_j = pca_result[j:end_j]

            distances = calc_chunk_distances(chunk_i, chunk_j, threshold)
            for x in range(distances.shape[0]):
                for y in range(distances.shape[1]):
                    if distances[x, y] > 0:
                        adjacency_matrix[i + x, j + y] = distances[x, y]
                        adjacency_matrix[j + y, i + x] = distances[x, y]
    return adjacency_matrix


@njit(parallel=True)
def calc_chunk_distances(chunk_i, chunk_j, threshold):
    """
    Calculate the distances between two chunks of data

    Args:
        chunk_i: data of the first chunk
        chunk_j: data of the second chunk
        threshold: trheshold for the distance calculation

    Returns:
        distances: matrix of distances between the two chunks
    """
    m, n = chunk_i.shape[0], chunk_j.shape[0]
    distances = np.zeros((m, n))
    for x in prange(m):
        for y in prange(n):
            dist = np.linalg.norm(chunk_i[x] - chunk_j[y])
            if dist > threshold:
                distances[x, y] = dist
    return distances


# =============================================================================
# Debugging area
# =============================================================================
# matrices = []
# data = reshape_matrices(matrices)
# del matrices
# pca_result = perform_pca(data)
# adjacency_matrix = calc_adjacency_matrix(pca_result)

# Created by gonzalezroy at 6/28/24
import time

import config as cfg
import mdtraj as md
import numpy as np
import topo_traj as tt
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def calc_gc(n_res, n_frames, R_locators, R, C_i):
    """
    Compute the mutual information matrix

    Args:
        n_res: number of residues
        n_frames: number of frames
        R_locators: indices of the coordinates of the residues in R matrix
        R: matrix of fluctuations
        C_i: array of individual residues fluctuations

    Returns:
        MI: mutual information matrix
    """
    k = -2 / 3
    MI = np.zeros((n_res, n_res))
    GC = np.zeros((n_res, n_res))
    for i in prange(n_res):
        for j in range(i + 1, n_res):
            ij_loc = np.concatenate((R_locators[i], R_locators[j]))
            dR = R[ij_loc]

            numerator = np.linalg.det(np.dot(dR, dR.T) / n_frames)
            denominator = C_i[i] * C_i[j]
            mutual_info = -0.5 * np.log(numerator / denominator)
            gen_corr = np.sqrt(1 - np.exp(k * mutual_info))

            MI[i, j] = mutual_info
            GC[i, j] = gen_corr
    return MI, GC


@njit(parallel=True, fastmath=True)
def calc_c_i(n_res, n_frames, R_locators, R):
    """
    Compute the individual residue fluctuations
    Args:
        n_res: number of residues
        n_frames: number of frames
        R_locators: indices of the coordinates of the residues in R matrix
        R: matrix of fluctuations

    Returns:
        C_i: array of individual residues fluctuations
    """
    C_i = np.zeros(n_res)
    for i in prange(n_res):
        i_indices = R_locators[i]
        dR = R[i_indices]
        C_i[i] = np.linalg.det(np.dot(dR, dR.T) / n_frames)
    return C_i


def get_locators(n_res, n_coords):
    """
    Get the locators of the coordinates of the residues in the R matrix

    Args:
        n_res: number of residues
        n_coords: number of coordinates

    Returns:
    """
    starts = range(0, n_res * n_coords, 3)
    ends = range(3, n_res * n_coords + 3, 3)
    locators = {i: np.asarray(range(starts[i], ends[i])) for i in range(n_res)}
    return locators


# =============================================================================
# Debugging and testing
# =============================================================================
time_here = time.time()

# Load trajectory
config_path = "/home/gonzalezroy/RoyHub/Code_pronucompass/example/params.cfg"
arg, dict_arg = cfg.parse_params(config_path)
full_traj = md.load(arg.traj, top=arg.topo)

# Indices of residues in the load trajectory and equivalence
resids_to_atoms, internal_equiv = tt.get_resids_indices(full_traj)
raw = {y: x for x in resids_to_atoms for y in resids_to_atoms[x]}
atoms_to_resids = tt.pydict_to_numbadict(raw)

# Superpose traj to the first one using only the alpha carbons & NA equivalents
_indices = tt.get_calpha_p_indices(full_traj, atoms_to_resids, numba=False)
alphaNA_idx = list(_indices.keys())
full_traj.superpose(full_traj, 0, atom_indices=alphaNA_idx)

# Coordinates in Angstroms
full_traj_xyz = full_traj.xyz
del full_traj
coords = full_traj_xyz[:, alphaNA_idx] * 10

# Compute the R matrix of fluctuations
num_frames, num_res, num_coords = coords.shape
fluctuations = coords - coords.mean(axis=0)
R = fluctuations.reshape(num_frames, num_res * num_coords).T
locators = tt.pydict_to_numbadict(get_locators(num_res, num_coords))

# Compute GC
C_i = calc_c_i(num_res, num_frames, locators, R)
_, _ = calc_gc(2, 2, locators, R[:2, :2], C_i)
MI, GC = calc_gc(num_res, num_frames, locators, R, C_i)
print(f"Time elapsed: {time.time() - time_here:.2f} s")
# fill lower triangle of symmetric GC

GC[np.tril_indices(num_res)] = GC.T[np.tril_indices(num_res)]

import correlations as corr

corr.plot_matrix(GC, "GC")

from collections import Counter

# =============================================================================
#
# =============================================================================
import hdbscan
import matplotlib.pyplot as plt

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=5, gen_min_span_tree=True, approx_min_span_tree=False
)
clusterer.fit(GC.astype(np.float64))
clusterer.labels_.max()

clusters = clusterer.labels_
persistences = clusterer.cluster_persistence_
probabilities = clusterer.probabilities_
outlier_score = clusterer.outlier_scores_
exemplars = clusterer.exemplars_

counts = Counter(clusters).most_common()

for x in counts:
    print(x)
    print(
        "residue " + " ".join([str(x) for x in np.where(clusterer.labels_ == x[0])[0]])
    )

clusterer.condensed_tree_.plot(select_clusters=True)
plt.savefig("condensed_tree.png")
plt.close()

clusterer.minimum_spanning_tree_.plot()
plt.savefig("minimum_spanning_tree.png")
plt.close()

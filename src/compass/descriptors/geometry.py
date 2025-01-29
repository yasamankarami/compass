# Created by gonzalezroy at 6/17/24
"""
Functions related to the calculation of geometric descriptors
"""
import os.path
import time
from os.path import join

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from numba import njit

import compass.descriptors.topo_traj as tt


@njit(parallel=False)
def calc_dist(atom1_coords, atom2_coords):
    """
    Computes the Euclidean distance between two atoms in a molecule

    Args:
        atom1_coords: 3D coordinate array of the first atom
        atom2_coords: 3D coordinate array of the second atom

    Returns:
        float: the Euclidean distance between the two atoms
    """
    dx = atom1_coords[0] - atom2_coords[0]
    dy = atom1_coords[1] - atom2_coords[1]
    dz = atom1_coords[2] - atom2_coords[2]
    return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)


@njit(parallel=False)
def calc_min_dist(coords1, coords2):
    """
    Get the minimumm distance between two sets of coordinates

    Args:
        coords1: coordinates of the first residue
        coords2: coordinates of the second residue

    Returns:
        The minimum distance between two sets of coordinates
    """
    # Constants
    n1 = coords1.shape[0]
    n2 = coords2.shape[0]

    # Find minimum distance using square values to save time
    min_dist_squared = np.inf
    for i in range(n1):
        for j in range(n2):
            dist_squared = (
                    (coords1[i][0] - coords2[j][0]) ** 2
                    + (coords1[i][1] - coords2[j][1]) ** 2
                    + (coords1[i][2] - coords2[j][2]) ** 2
            )
            if dist_squared < min_dist_squared:
                min_dist_squared = dist_squared
    return np.sqrt(min_dist_squared)


@njit(parallel=False)
def calc_single_angle(d, h, a):
    """
    Computes the angle between three atoms

    Args:
        d (donor): Coordinates of the first atom (x, y, z)
        h (hydrogen): Coordinates of the second atom (x, y, z).
        a (acceptor): Coordinates of the third atom (x, y, z).

    Returns:
        angle_deg: the angle in degrees
    """
    # Compute vectors
    dh = d - h
    ah = a - h

    # Compute dot & norms
    dot_product = np.dot(dh, ah)
    dh_norm = np.linalg.norm(dh)
    ah_norm = np.linalg.norm(ah)

    # Compute angle
    angle_rad = np.arccos(dot_product / (dh_norm * ah_norm))
    angle_deg = np.rad2deg(angle_rad)
    return angle_deg


@njit(parallel=False)
def find_sb(frame_coords, oxy_i, nitro_j, k):
    """
    Find a single salt bridge between two residues

    Args:
        frame_coords: 3D coordinates of the frame
        oxy_i: oxygen atom index of the first residue
        nitro_j: nitrogen atom index of the second residue
        k: cutoff distance

    Returns:
        bool: True if a salt bridge is found, False otherwise
    """
    # Constants
    n1 = oxy_i.size
    n2 = nitro_j.size
    min_dist_squared = k ** k

    # Find minimum distance using square values to save time
    for i in range(n1):
        coords1 = frame_coords[oxy_i[i]]
        for j in range(n2):
            coords2 = frame_coords[nitro_j[j]]

            dist_squared = (
                    (coords1[0] - coords2[0]) ** 2
                    + (coords1[1] - coords2[1]) ** 2
                    + (coords1[2] - coords2[2]) ** 2
            )
            if dist_squared < min_dist_squared:
                return True
    return False


@njit(parallel=False)
def find_hb(frame_coords, donors_i, hydros_i, acceptors_j, da_cut, ha_cut,
            dha_cut):
    """
    Find a single hydrogen bond between two residues
    Args:
        frame_coords: 3D coordinates of the frame
        donors_i: donor atom indices of the first residue
        hydros_i: hydrogen atom indices of the first residue
        acceptors_j: acceptor atom indices of the second residue
        da_cut: distance cutoff for the donor-acceptor distance
        ha_cut: distance cutoff for the hydrogen-acceptor distance
        dha_cut: angle cutoff for the donor-hydrogen-acceptor angle

    Returns:
        bool: True if a hydrogen bond is found, False otherwise
    """
    # Constants
    n1 = donors_i.size
    n2 = acceptors_j.size

    # Find a suitable angle
    for i in range(n1):
        coords_d = frame_coords[donors_i[i]]
        coords_h = frame_coords[hydros_i[i]]
        for j in range(n2):
            coords_a = frame_coords[acceptors_j[j]]

            da_dist = calc_dist(coords_d, coords_a)
            if da_dist < da_cut:
                ha_dist = calc_dist(coords_h, coords_a)
                if ha_dist < ha_cut:
                    angle = calc_single_angle(coords_d, coords_h, coords_a)
                    if angle > dha_cut:
                        return True
    return False


def save_matrix(arr, n, out_name, norm=False, prec=2):
    """
    Save a matrix to a file

    Args:
        arr: array to save
        n: number of columns
        missing: missing indices
        out_name: output file name
        norm: normalize the matrix?
        diag: fill the diagonal with 1?
        prec: precision of the values to save

    Returns:
        matrix_name: name of the saved matrix
    """
    # Convert to matrix if needed
    matrix = tt.to_matrix(arr, n) if len(arr.shape) == 1 else arr

    # Normalize if requested
    if norm:
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        matrix = (matrix - min_val) / (max_val - min_val)

    # Fill diagonal if requested
    # if diag:
    #     np.fill_diagonal(matrix, 1.0)

    # Fill undefined rows with 0
    # matrix[missing, :] = matrix[:, missing] = 0

    # Save matrix
    np.savetxt(out_name, matrix, fmt=f"%.{prec}f")
    return matrix


def get_matrix_name(out_dir, title, suffix):
    """
    Generate a matrix name

    Args:
        out_dir: output directory
        title: title of the matrix
        suffix: suffix of the matrix

    Returns:
        matrix_name: name of the matrix
    """
    matrix_dir = join(out_dir, "matrices")
    if not os.path.exists(matrix_dir):
        os.makedirs(matrix_dir, exist_ok=True)
    return join(out_dir, 'matrices', f"{title}_{suffix}.mat")


def plot_matrix(matrix, matrix_title, output_name):
    """
    Plot the Generalized Correlation matrix.

    Args:
        matrix: Generalized Correlation matrix
        matrix_title: title of the matrix
        output_name: output name for the plot
    """
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(matrix, cmap="jet")
    plt.title(matrix_title)
    plt.xlabel("Residue Index")
    plt.ylabel("Residue Index")
    plt.savefig(output_name)
    plt.close()


def process_matrices(arg, n, calphas, ave_min_dist, occ_nb, cp, occ_sb, occ_hb,
                     occ_int, mi, gc, first_timer, ):
    # Declare missing residues
    # cp_miss = [i for i, x in enumerate(calphas) if calphas[x] == -1]

    # Declare matrices to process
    matrices = {
        "MINDIST": {"data": ave_min_dist, "norm": False,
                    "prec": 4, "title": "Pairwise Minimum Distances"},

        "NONBOND": {"data": occ_nb, "norm": False, "prec": 4,
                    "title": "Non-Bonded Interactions"},

        "SALTBRIDGES": {"data": occ_sb, "norm": False, "prec": 4,
                        "title": "Salt Bridges"},

        "HBONDS": {"data": occ_hb, "norm": False, "prec": 4,
                   "title": "Hydrogen Bonds"},

        "INTERACTIONS": {"data": occ_int, "norm": False, "prec": 2,
                         "title": "Interactions"},

        "COMMPROP": {"data": cp, "norm": True, "prec": 4,
                     "title": "Communication Properties"},

        "MI": {"data": mi, "norm": True, "prec": 4,
               "title": "Mutual Information"},

        "GC": {"data": gc, "norm": True, "prec": 4,
               "title": "Generalized Correlation"},
    }

    # Process matrices
    matrices_name = {}
    for matrix in matrices:
        # Get matrix data
        data = matrices[matrix]["data"]
        # miss_list = matrices[matrix]["miss"]
        normalize = matrices[matrix]["norm"]
        precision = matrices[matrix]["prec"]

        # Save matrices
        matrix_name = get_matrix_name(arg.out_dir, arg.title, matrix)
        matrices_name.update({matrix: matrix_name})
        matrix_data = save_matrix(data, n, matrix_name, norm=normalize,
                                  prec=precision)
        matrices[matrix].update({"data": matrix_data})

        # Plot matrices
        plot_name = matrix_name.replace(".mat", ".png")
        matrix_title = matrices[matrix]["title"]
        plot_matrix(matrix_data, matrix_title, plot_name)

    saving_time = round(time.time() - first_timer, 2)
    print(f" ⏱️  Until saving & plotting matrices: {saving_time} s")
    return matrices, matrices_name

#
# # todo: specify correct diagonal filling behaviour
# # todo: correct the titles of the graph
#
# no_miss = []
# cp_miss = [i for i, x in enumerate(calphas) if calphas[x] == -1]
#
# min_dist_name = geom.get_matrix_name(arg.out_dir, arg.title, "MINDIST")
# dist_mat = geom.save_matrix(ave_min_dist, n, no_miss, min_dist_name)
# del ave_min_dist
# corr.plot_matrix(dist_mat, min_dist_name.replace('.mat', '.png'))
#
# nb_name = geom.get_matrix_name(arg.out_dir, arg.title, "NONBOND")
# nb_mat = geom.save_matrix(occ_nb, n, no_miss, nb_name)
# del occ_nb
# corr.plot_matrix(nb_mat, nb_name.replace('.mat', '.png'))
#
# cp_name = geom.get_matrix_name(arg.out_dir, arg.title, "COMMPROP")
# cp_mat = geom.save_matrix(cp, n, cp_miss, cp_name, norm=True, prec=6)
# del cp
# corr.plot_matrix(cp_mat, cp_name.replace('.mat', '.png'))
#
# sb_name = geom.get_matrix_name(arg.out_dir, arg.title, "SALTBRIDGES")
# sb_mat = geom.save_matrix(occ_sb, n, no_miss, sb_name)
# del occ_sb
# corr.plot_matrix(sb_mat, sb_name.replace('.mat', '.png'))
#
# hb_name = geom.get_matrix_name(arg.out_dir, arg.title, "HBONDS")
# hb_mat = geom.save_matrix(occ_hb, n, no_miss, hb_name)
# del occ_hb
# corr.plot_matrix(hb_mat, hb_name.replace('.mat', '.png'))
#
# inter_name = geom.get_matrix_name(arg.out_dir, arg.title, "INTERACTIONS")
# int_mat = geom.save_matrix(occ_int, n, no_miss, inter_name)
# del occ_int
# corr.plot_matrix(int_mat, inter_name.replace('.mat', '.png'))
#
# mi_name = geom.get_matrix_name(arg.out_dir, arg.title, "MI")
# mi_mat = geom.save_matrix(mi, n, cp_miss, mi_name)
# del mi
# corr.plot_matrix(mi_mat, mi_name.replace('.mat', '.png'))
#
# gc_name = geom.get_matrix_name(arg.out_dir, arg.title, "GC")
# gc_mat = geom.save_matrix(gc, n, cp_miss, gc_name)
# del gc
# corr.plot_matrix(gc_mat, gc_name.replace('.mat', '.png'))
#
# saving_time = round(time.time() - first_timer, 2)
# print(f'Until saving & plotting matrices: {saving_time} s')

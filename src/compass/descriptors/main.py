# Created by gonzalezroy at 6/17/24
"""
Functions related to the calculation of geometric descriptor matrices
"""
import time

import numpy as np
from numba import njit, prange

import compass.descriptors.correlations as corr
import compass.descriptors.geometry as geom
import compass.descriptors.topo_traj as tt


# todo: update the docstrings


def compute_descriptors(mini_traj, trajs, arg, resids_to_atoms, resids_to_noh,
                        calphas, oxy, nitro, donors, hydros, acceptors,
                        corr_indices, first_timer):
    """
    Compute the compass descriptors for the trajectory

    Args:
        mini_traj: first frame of the trajectory
        trajs: trajectory or list of trajectories
        arg: namespace containing the arguments
        resids_to_atoms: mapping of residues indices to the atoms indices
        resids_to_noh: mapping of residues indices to the noh atoms indices
        calphas: indices of the calpha atoms
        oxy: indices of the oxygen atoms
        nitro: indices of the nitrogen atoms
        donors: indices of the donor atoms
        hydros: indices of the hydrogen atoms
        acceptors: indices of the acceptor atoms
        corr_indices: indices of the atoms to compute the correlation matrices
         (calpha for protein and c5' for nucleic acids)
        first_timer: first time stamp

    Returns:

    """
    # Initialize containers
    n_resids = len(resids_to_atoms)
    n_pairs = int(n_resids * (n_resids - 1) / 2)
    pair_min_dist_sum = np.zeros(n_pairs)
    pair_cp_sum = np.zeros(n_pairs)
    pair_nb_sum = np.zeros(n_pairs)
    pair_sb_sum = np.zeros(n_pairs)
    pair_hb_sum = np.zeros(n_pairs)
    pair_int_sum = np.zeros(n_pairs)

    # Compile numba function
    get_chunk_info(mini_traj.xyz, resids_to_atoms, resids_to_noh, arg.nb_cut,
                   arg.sb_cut, arg.da_cut, arg.ha_cut, arg.dha_cut, calphas,
                   oxy, nitro, donors, hydros, acceptors)

    comp_time = round(time.time() - first_timer, 2)
    print(
        f" ⏱️  Until compilation of descriptors-related functions: {comp_time} s")

    # Do a first pass to compute most descriptors
    chunks = tt.get_xyz_chunks(trajs, arg.topo, chunk_size=100)
    # print(np.shape(trajs))
    n_frames = 0
    for chunk in chunks:
        n_frames += chunk.shape[0]
        pair_min_dist, pair_cp, pair_nb, pair_sb, pair_hb, pair_int = \
            get_chunk_info(chunk, resids_to_atoms, resids_to_noh, arg.nb_cut,
                           arg.sb_cut, arg.da_cut, arg.ha_cut, arg.dha_cut,
                           calphas, oxy, nitro, donors, hydros, acceptors)

        pair_min_dist_sum = sum_arrays(pair_min_dist, pair_min_dist_sum)
        pair_cp_sum = sum_arrays(pair_cp, pair_cp_sum)
        pair_nb_sum = sum_arrays(pair_nb, pair_nb_sum)
        pair_sb_sum = sum_arrays(pair_sb, pair_sb_sum)
        pair_hb_sum = sum_arrays(pair_hb, pair_hb_sum)
        pair_int_sum = sum_arrays(pair_int, pair_int_sum)

    # Compute average values
    ave_min_dist = (pair_min_dist_sum / n_frames) * 10
    ave_pair_cp = pair_cp_sum / n_frames
    occ_nb = pair_nb_sum / n_frames
    occ_sb = pair_sb_sum / n_frames
    occ_hb = pair_hb_sum / n_frames
    occ_int = pair_int_sum / n_frames

    # Do a 2nd pass to compute cp & extract coords for correlation matrices
    pair_cp_sum2 = np.zeros(n_pairs)
    chunks = tt.get_xyz_chunks(trajs, arg.topo, chunk_size=100)
    corr_coords = np.zeros((n_frames, len(corr_indices), 3))

    k = 0
    get_chunk_cp(mini_traj.xyz, resids_to_atoms, ave_pair_cp, calphas)
    for chunk in chunks:
        # Compute CP
        pair_cp2 = get_chunk_cp(chunk, resids_to_atoms, ave_pair_cp, calphas)
        pair_cp_sum2 = sum_arrays(pair_cp2, pair_cp_sum2)

        # Get correlation coordinates
        corr_chunk = chunk[:, corr_indices]
        corr_coords[k: k + corr_chunk.shape[0], :] = corr_chunk
        k += corr_chunk.shape[0]
    cp = pair_cp_sum2 / n_frames * 100

    # Compute MI & GC
    mi, gc = corr.compute_gc_matrix(corr_coords, num_atoms_per_residue=3)

    running_time = round(time.time() - first_timer, 2)
    print(f" 📋 System details: number of frames are {n_frames}")
    print(f" ⏱️  Until descriptors computed: {running_time} s")
    return ave_min_dist, occ_nb, cp, occ_sb, occ_hb, occ_int, mi, gc


@njit(parallel=True)
def get_chunk_info(traj_coords, resids_to_atoms, resids_to_noh, nb_cut, sb_cut,
                   da_cut, ha_cut, dha_cut, calphas, oxy, nitro, donors,
                   hydros, acceptors):
    """
    Get the minimum distance between every pair of residues averaged along
    the trajectory

    Args:
        traj_coords: xyz coordinates of the trajectory
        resids_to_atoms: dict mapping residues indices to the atoms indices
        resids_to_noh: dict mapping residues indices to the noh atoms indices
        nb_cut: distance cutoff for non-bonded contacts calculation
        sb_cut: distance cutoff for salt bridges calculation
        da_cut: distance cutoff for DA in hydrogen bonds calculation
        ha_cut: distance cutoff for HA in hydrogen bonds calculation
        dha_cut: angle cutoff for DHA in hydrogen bonds calculation
        calphas: dict mapping residues indices to their calpha atoms indices
        oxy: dict mapping residues indices to their oxygen atoms indices
        nitro: dict mapping residues indices to their nitrogen atoms indices
        donors: dict mapping residues indices to the donor atoms indices
        hydros: dict mapping residues indices to the hydrogen atoms indices
        acceptors: dict mapping residues indices to the acceptor atoms indices

    Returns:
        ave_pair_min_dist: average along the trajectory of the minimum distance
                           between every pair of residues
        percent_nb: percent of non-bonded contacts occupancy between every
                         pair of residues
    """
    # Constants
    n_resids = len(resids_to_atoms)
    n_pairs = int(n_resids * (n_resids - 1) / 2)
    n_frames = len(traj_coords)

    # Initialize containers
    pair_min_dist_sum = np.zeros(n_pairs)
    pair_cp_sum = np.zeros(n_pairs)
    pair_nb_sum = np.zeros(n_pairs)
    pair_sb_sum = np.zeros(n_pairs)
    pair_hb_sum = np.zeros(n_pairs)
    pair_int_sum = np.zeros(n_pairs)
    # print(f" 📋 System details: Number of frames are {n_frames}, number of backbone atoms are {n_resids}")
    # Compute all interactions for each frame in parallel
    for frame in prange(n_frames):
        frame_coords = traj_coords[frame]
        pair_min_dists, pair_nb, pair_cp, pair_sb, pair_hb, pair_int = \
            get_frame_info(frame_coords, resids_to_atoms, resids_to_noh,
                           nb_cut, sb_cut, da_cut, ha_cut, dha_cut, calphas,
                           oxy, nitro, donors, hydros, acceptors, )

        # Uptade the sum of interactions
        pair_min_dist_sum += pair_min_dists
        pair_cp_sum += pair_cp
        pair_nb_sum += pair_nb
        pair_sb_sum += pair_sb
        pair_hb_sum += pair_hb
        pair_int_sum += pair_int
    return (
    pair_min_dist_sum, pair_cp_sum, pair_nb_sum, pair_sb_sum, pair_hb_sum,
    pair_int_sum)


@njit(parallel=True)
def get_chunk_cp(traj_coords, resids_to_atoms, pair_cp_sum, calphas):
    """
    Get the minimum distance between every pair of residues averaged along
    the trajectory

    Args:
        traj_coords: xyz coordinates of the trajectory
        resids_to_atoms: dict mapping residues indices to the atoms indices
        calphas: dict mapping residues indices to their calpha atoms indices
        pair_cp_sum: pairwise sum of the distances between calpha atoms

    Returns:
        ave_pair_min_dist: average along the trajectory of the minimum distance
                           between every pair of residues
        percent_nb: percent of non-bonded contacts occupancy between every
                         pair of residues
    """
    # Constants
    n_resids = len(resids_to_atoms)
    n_pairs = int(n_resids * (n_resids - 1) / 2)
    n_frames = len(traj_coords)
    # print(n_resids, n_pairs, n_frames, "get_chunk_cp in main")

    # Get the cp in a second pass to avoid RAM issues
    ave_pair_cp = pair_cp_sum / n_frames
    cp_values = np.zeros(n_pairs)
    for frame in prange(n_frames):
        k = 0
        triangle = np.zeros(n_pairs, dtype=float)
        frame_coords = traj_coords[frame]

        for i in range(n_resids):
            calpha_i = frame_coords[calphas[i]]
            for j in range(i + 1, n_resids):
                calpha_j = frame_coords[calphas[j]]
                d_ij = geom.calc_dist(calpha_i, calpha_j)
                triangle[k] = d_ij
                k += 1
        cp_values += (triangle - ave_pair_cp) ** 2
    return cp_values


@njit(parallel=False)
def get_frame_info(frame_coords, resids_to_atoms, resids_to_noh, nb_cut,
                   sb_cut, da_cut, ha_cut, dha_cut, calphas, oxy, nitro,
                   donors, hydros, acceptors):
    """
    Args:
        frame_coords: xyz coordinates of the frame
        resids_to_atoms: dict mapping residues indices to the atoms indices
        resids_to_noh: dict mapping residues indices to the noh atoms indices
        nb_cut: distance cutoff for non-bonded contacts calculation
        sb_cut: distance cutoff for salt bridges calculation
        da_cut: distance cutoff for DA in hydrogen bonds calculation
        ha_cut: distance cutoff for HA in hydrogen bonds calculation
        dha_cut: angle cutoff for DHA in hydrogen bonds calculation
        calphas: dict mapping residues indices to their calpha atoms indices
        oxy: dict mapping residues indices to their oxygen atoms indices
        nitro: dict mapping residues indices to their nitrogen atoms indices
        donors: dict mapping residues indices to the donor atoms indices
        hydros: dict mapping residues indices to the hydrogen atoms indices
        acceptors: dict mapping residues indices to the acceptor atoms indices

    Returns:
        pair_min_dists: minimum distance between every pair of residues
        pair_nb: non-bonded contacts between every pair of residues
        pair_cp: distance between calpha atoms of every pair of residues
        pair_sb: salt bridges between every pair of residues
        pair_hb: hydrogen bonds between every pair of residues
        pair_int: interactions between every pair of residues
    """
    # Constants
    n_resids = len(resids_to_atoms)
    n_pairs = int(n_resids * (n_resids - 1) / 2)

    # Initialize containers
    pair_min_dists = np.zeros(n_pairs)
    pair_nb = np.zeros(n_pairs)
    pair_cp = np.zeros(n_pairs)
    pair_sb = np.zeros(n_pairs)
    pair_hb = np.zeros(n_pairs)
    pair_int = np.zeros(n_pairs)

    # Get min dist for all residues in frame
    index = 0
    for i in prange(n_resids):
        # coords_i = frame_coords[resids_to_atoms[i]]
        coords_i = frame_coords[resids_to_noh[i]]
        calpha_i = frame_coords[calphas[i]]
        for j in range(i + 1, n_resids):
            # coords_j = frame_coords[resids_to_atoms[j]]
            coords_j = frame_coords[resids_to_noh[j]]
            calpha_j = frame_coords[calphas[j]]

            # MINDIST: Get min distance between residues i and j
            min_dist = geom.calc_min_dist(coords_i, coords_j)
            pair_min_dists[index] = min_dist

            # NONBOND: Get one non-bonded contact between residues i and j
            if min_dist < nb_cut:
                pair_nb[index] = 1

            # COMMPROP: Get the distance between calpha atoms
            dist_calpha = geom.calc_dist(calpha_i, calpha_j)
            pair_cp[index] = dist_calpha

            # SALTBRIDGES: Get one salt bridge between residues i and j
            sb = 0
            if (min_dist < sb_cut) and (i + 1 != j):

                # Direct case
                oxy_i = tt.dict_get(oxy, i)
                nitro_j = tt.dict_get(nitro, j)
                if (oxy_i is not None) and (nitro_j is not None):
                    sb += geom.find_sb(frame_coords, oxy_i, nitro_j, sb_cut)

                # Inverse case
                oxy_j = tt.dict_get(oxy, j)
                nitro_i = tt.dict_get(nitro, i)
                if (oxy_j is not None) and (nitro_i is not None):
                    sb += geom.find_sb(frame_coords, oxy_j, nitro_i, sb_cut)
            if sb:
                pair_sb[index] = 1

            # HBONDS: Get one hydrogen bond between residues i and j
            hb = 0
            if min_dist <= da_cut:

                # Direct case
                donors_i = tt.dict_get(donors, i)
                hydros_i = tt.dict_get(hydros, i)
                acceptors_j = tt.dict_get(acceptors, j)
                if (donors_i is not None) and (acceptors_j is not None):
                    hb += geom.find_hb(frame_coords, donors_i, hydros_i,
                                       acceptors_j, da_cut, ha_cut, dha_cut)

                # Inverse case
                donors_j = tt.dict_get(donors, j)
                hydros_j = tt.dict_get(hydros, j)
                acceptors_i = tt.dict_get(acceptors, i)
                if (donors_j is not None) and (acceptors_i is not None):
                    hb += geom.find_hb(frame_coords, donors_j, hydros_j,
                                       acceptors_i, da_cut, ha_cut, dha_cut)
            if hb:
                pair_hb[index] = 1

            # INTERACTIONS: Get one interaction between residues i and j
            if sb or hb:
                pair_int[index] = 1

            index += 1
    return pair_min_dists, pair_nb, pair_cp, pair_sb, pair_hb, pair_int


@njit(parallel=True)
def sum_arrays(arr1, arr2):
    """
    Sum two one-dimensional arrays

    Args:
        arr1: first array
        arr2: second array

    Returns:
        sum_arr: sum of the two arrays
    """
    sum_arr = np.zeros(len(arr1))
    for i in prange(len(arr1)):
        sum_arr[i] = arr1[i] + arr2[i]
    return sum_arr

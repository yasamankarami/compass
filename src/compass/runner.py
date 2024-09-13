# Created by gonzalezroy at 6/6/24
"""
Run the computation of descriptors
"""
# todo: chunkify processing
# todo: integrate the ability to load replicas
# todo: change P atoms as NA equivalents to Calpha (see N1/N9)
# todo: multiply by 10 the coordinates to convert nm to Angstroms

# todo: superpose trajectory to the first frame using Calpha atoms for GC
# todo: noH by default in the min dist calculation

import time
import sys
import compass.descriptors.config as cfg
import compass.descriptors.geometry as geom
import compass.descriptors.main as mm
import compass.descriptors.pca as pca
import compass.descriptors.topo_traj as tt


def runner():
    """
    Entry point for running the computation of descriptors
    """
    first_timer = time.time()
    # =============================================================================
    # 1. Prelude
    # =============================================================================
    # ==== Parse configuration file
    if len(sys.argv) != 2:
        raise ValueError(
            '\ncompass syntax is: compass path-to-config-file')
    config_path = sys.argv[1]
    # config_path = "./example/params.cfg"
    arg, dict_arg = cfg.parse_params(config_path)

    # ==== Prepare datastructures & containers
    (mini_traj, trajs, resids_to_atoms, resids_to_noh, calphas, oxy, nitro,
     donors, hydros, acceptors, corr_indices) = tt.prepare_datastructures(
        arg, first_timer)

    # =============================================================================
    # 2. Computing
    # =============================================================================
    ave_min_dist, occ_nb, cp, occ_sb, occ_hb, occ_int, mi, gc = mm.compute_descriptors(
        mini_traj, trajs, arg, resids_to_atoms, resids_to_noh, calphas, oxy,
        nitro, donors, hydros, acceptors, corr_indices, first_timer)

    # =============================================================================
    # 3. Saving matrices
    # =============================================================================
    n = len(resids_to_atoms)
    matrices = geom.process_matrices(arg, n, calphas, ave_min_dist, occ_nb, cp,
                                     occ_sb, occ_hb, occ_int, mi, gc,
                                     first_timer)

    # =============================================================================
    # 4. Perform PCA & generate adjacency matrix from PCA results
    # =============================================================================

    # Select the matrices to be used in the PCA
    gc_mat = matrices["GC"]["data"]
    int_mat = matrices["INTERACTIONS"]["data"]
    cp_mat = matrices["COMMPROP"]["data"]
    dist_mat = matrices["MINDIST"]["data"]
    matrices = [gc_mat, int_mat, cp_mat, dist_mat]
    data = pca.reshape_matrices(matrices)
    del matrices

    # Perform PCA & generate adjacency matrix
    pca_result = pca.perform_pca(data)
    del data
    adj_mat_raw = pca.calc_adjacency_matrix(pca_result)
    adj_mat = adj_mat_raw.toarray()
    del adj_mat_raw

    adj_name = geom.get_matrix_name(arg.out_dir, arg.title, "ADJACENCY")
    adj_mat = geom.save_matrix(adj_mat, n, [], adj_name)
    geom.plot_matrix(adj_mat, adj_name.replace(".mat", ".png"))

    pca_time = round(time.time() - first_timer, 2)
    print(f"Until PCA & Adjacency matrix computing: {pca_time} s")
    print(f"Normal Termination")

# Created by gonzalezroy at 6/6/24
"""
Run the computation of descriptors
"""
# done: change P atoms as NA equivalents to Calpha -> C5'
# done: noH by default in the min dist calculation
# todo: remove the Debugging Area
# todo: compare vs comma

# %%
import os
import sys
import time
from os.path import join

import compass.descriptors.config as cfg
import compass.descriptors.geometry as geom
import compass.descriptors.main as mm
import compass.descriptors.pca as pca
import compass.descriptors.topo_traj as tt
import compass.network.generals as gn


# %%

def runner():
    """
    Entry point for running the compass workflow
    """
    # %%=======================================================================
    # 1. Prelude
    # =========================================================================
    # Parse configuration file
    if len(sys.argv) != 2:
        raise ValueError(
            '\ncompass syntax is: compass path-to-config-file')
    config_path = sys.argv[1]
    # config_path = "/home/rglez/RoyHub/compass/example/params.cfg"
    first_timer = time.time()
    arg, dict_arg = cfg.parse_params(config_path)

    # Prepare datastructures & containers
    (mini_traj, trajs, resids_to_atoms, resids_to_noh, calphas, oxy, nitro,
     donors, hydros, acceptors, corr_indices) = tt.prepare_datastructures(
        arg, first_timer)

    # %%=======================================================================
    # 2. Compute descriptors
    # =========================================================================
    ave_min_dist, occ_nb, cp, occ_sb, occ_hb, occ_int, mi, gc = \
        mm.compute_descriptors(mini_traj, trajs, arg, resids_to_atoms,
                               resids_to_noh, calphas, oxy, nitro, donors,
                               hydros, acceptors, corr_indices, first_timer)
    # invert CP matrix
    cp = abs(cp - max(cp))
    # %%=======================================================================
    # 3. Save matrices
    # =========================================================================
    n = len(resids_to_atoms)
    matrices, matrices_names = \
        geom.process_matrices(arg, n, calphas, ave_min_dist, occ_nb, cp,
                              occ_sb, occ_hb, occ_int, mi, gc, first_timer)

    # %%=======================================================================
    # 4. Perform PCA & generate adjacency matrix from PCA results
    # =========================================================================

    # Select the matrices to be used in the PCA
    adj_name = pca.run_pca(arg, matrices, n, first_timer)

    # %%=======================================================================
    # 5. Perform network analyses
    # =========================================================================

    # Construct graphs
    arg.adjacency_file = adj_name
    arg.min_dist_matrix_file = matrices_names["MINDIST"]
    arg.pdb_file_path = dict_arg["generals"]["topology"]
    arg.network_dir = join(dict_arg["generals"]["output_dir"], 'network')
    os.makedirs(arg.network_dir, exist_ok=True)
    gn.process_graphs(arg)
    print(
        f'\nUntil graphs construction: {round(time.time() - first_timer, 2)} s')

    # Compute network parameters
    gn.process_graph_files(arg.network_dir)
    # Find alternative paths for a specific graph
    # graph, atom_mapping = rf.ReadFiles().load_graph_and_mapping('path_to_graph_file.json')
    # alternative_paths = network_parameters.find_alternative_paths(graph, atom_mapping, 'source_residue', 'target_residue')
    print(
        f'Until network parameters computed: {round(time.time() - first_timer, 2)} s')

    # Compute communities and cliques
    # method = input("Select community detection method ('leiden' or 'girvan'): ").strip().lower()
    method = 'leiden'
    gn.process_graph_files_for_communities_and_cliques(arg.network_dir, method)
    print(
        f'Until communities and cliques detection: {round(time.time() - first_timer, 2)} s')

    # =============================================================================
    # 6. Generate PyMOL scripts
    # =============================================================================1
    gn.generate_pymol_scripts(arg.network_dir, arg.pdb_file_path)
    print(
        f'Until pymol scripts generation: {round(time.time() - first_timer, 2)} s')
    print(f"Normal Termination")

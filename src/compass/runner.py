# Created by gonzalezroy at 6/6/24
"""
Run the computation of descriptors
"""

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


def runner():
    """
    Entry point for running the compass workflow
    """
    # =========================================================================
    # 1. Prelude
    # =========================================================================
    # Parse configuration file
    if len(sys.argv) != 2:
        raise ValueError(
            '\ncompass syntax is: compass path-to-config-file')
    config_path = sys.argv[1]
    first_timer = time.time()
    arg, dict_arg = cfg.parse_params(config_path)

    # Prepare data structures
    (mini_traj, trajs, resids_to_atoms, resids_to_noh, calphas, oxy, nitro,
     donors, hydros, acceptors, corr_indices) = tt.prepare_datastructures(
        arg, first_timer)

    # =========================================================================
    # 2. Compute descriptors
    # =========================================================================
    (ave_min_dist, occ_nb, cp, occ_sb, occ_hb, occ_int, mi,
     gc) = mm.compute_descriptors(mini_traj, trajs, arg, resids_to_atoms,
                                  resids_to_noh, calphas, oxy, nitro, donors,
                                  hydros, acceptors, corr_indices, first_timer)
    # invert CP matrix
    cp = abs(cp - max(cp))

    # =========================================================================
    # 3. Save matrices
    # =========================================================================
    n = len(resids_to_atoms)
    matrices, matrices_names = geom.process_matrices(
        arg, n, calphas, ave_min_dist, occ_nb, cp, occ_sb, occ_hb, occ_int, mi,
        gc, first_timer)

    # =========================================================================
    # 4. Perform PCA & generate adjacency matrix from PCA results
    # =========================================================================
    # Select the matrices to be used in the PCA
    adj_name = pca.run_pca(arg, matrices, n, first_timer)

    # =========================================================================
    # 5. Perform network analyses
    # =========================================================================
    # Construct graphs
    arg.adjacency_file = adj_name
    arg.min_dist_matrix_file = matrices_names["MINDIST"]
    arg.pdb_file_path = arg.topo
    arg.network_dir = join(dict_arg["generals"]["output_dir"], 'network')
    os.makedirs(arg.network_dir, exist_ok=True)
    dist_cutoffs = [dict_arg["distance cutoffs"]["Graph"],
                    dict_arg["distance cutoffs"]["Cliques"]]

    # print(dist_cutoffs)
    gn.process_graphs(arg, dist_cutoffs)
    graph_time = round(time.time() - first_timer, 2)
    print(f' ⏳  Until graphs construction: {graph_time} s')

    # Compute network parameters
    gn.process_graph_files(arg.network_dir, dist_cutoffs[0])
    network_time = round(time.time() - first_timer, 2)
    print(f' ⏳  Until network parameters computed: {network_time} s')

    # Compute communities and cliques
    gn.process_graph_files_for_communities_and_cliques(arg.network_dir,
                                                       dist_cutoffs[0],
                                                       dist_cutoffs[1])
    clique_time = round(time.time() - first_timer, 2)
    print(f' ⏳  Until communities and cliques detection: {clique_time} s')

    # =========================================================================
    # 6. Generate PyMOL scripts
    # =========================================================================
    gn.generate_pymol_scripts(arg.network_dir, arg.pdb_file_path,
                              dist_cutoffs[0], dist_cutoffs[1])
    if dict_arg["paths"]["find_path"] == 'True':
        source_residues = dict_arg["paths"]["sources"].split(",")
        target_residues = dict_arg["paths"]["targets"].split(",")

        for source_residue in source_residues:
            for target_residue in target_residues:
                gn.find_paths(arg.pdb_file_path, arg.network_dir,
                              dist_cutoffs[0], source_residue.strip(),
                              target_residue.strip())

    pymol_time = round(time.time() - first_timer, 2)
    print(f' ⏳  Until pymol scripts generation: {pymol_time} s')
    print(f"**** -------Normal Termination -------****")

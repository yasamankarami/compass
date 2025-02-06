# Created by rglez at 9/13/24
import os
import time
import shutil
import json
import networkx as nx
from compass.network.graph_constructor import GraphConstructor
from compass.network.networkparameters import NetworkParameters
from compass.network.read_files import ReadFiles
from compass.network.communities_cliques import CommunityDetector, CliqueDetector
from compass.network.pymol_visualizer import PyMOLVisualizer


def process_graphs(param_space, distance_cutoffs):
    """
    Constructs graphs based on distance cutoffs, processes them, and saves results.

    Args:
        param_space (Namespace): An object containing file paths and directories.
        distance_cutoffs (list, optional): List of distance cutoffs to use. If None, a default list is used.
    """
    # Initialize GraphConstructor
    graph_constructor = GraphConstructor(
        distance_file=param_space.min_dist_matrix_file,
        adjacency_file=param_space.adjacency_file,
        distance_cutoffs=distance_cutoffs
    )
    atom_mapping, _ = graph_constructor.reader.atom_mapping(param_space.pdb_file_path)

    # Iterate over each distance cutoff to build and process the graph
    for distance_cutoff in graph_constructor.distance_cutoffs:
        start_time = time.time()
        # Generate atom mapping before saving the graph
        # Build the graph
        G = graph_constructor.build_graph_from_matrices(
            distance_file=param_space.min_dist_matrix_file,
            adjacency_file=param_space.adjacency_file,
            distance_cutoff=distance_cutoff,
            atom_mapping = atom_mapping)
        # Ensure graph connectivity
        G = graph_constructor.ensure_graph_connectivity(G)
        # Define file names based on the distance cutoff
        graph_filename = f"graph_cutoff_{distance_cutoff}.json"
        output_graph_file = os.path.join(param_space.network_dir, graph_filename)
        # Save the graph and atom mapping
        graph_constructor.save_graph_and_mapping(G, atom_mapping, output_file=output_graph_file)
        # Plot and save histogram
        output_file_prefix = os.path.join(param_space.network_dir, f"graph_cutoff_{distance_cutoff}")
        graph_constructor.plot_and_save_histogram(G, output_file_prefix=output_file_prefix)
        print(f" 🖥️  Processed graph for cutoff {distance_cutoff} in {round(time.time() - start_time, 2)} seconds")

def process_graph_files(results_dir, dist_cutoff_graph):
    """
    Process all graph files in the specified results directory.

    Args:
        results_dir (str): Path to the directory containing the result files.
    """
    # List all .json files in the results directory
    for filename in os.listdir(results_dir):
        if filename.endswith('.json') and filename.startswith('graph_cutoff_'+ dist_cutoff_graph):
            # Construct the full path to the JSON file
            json_path = os.path.join(results_dir, filename)
            # Load the graph and atom mapping
            graph, atom_mapping = ReadFiles().load_graph_and_mapping(json_path)
            # Initialize NetworkParameters
            network_parameters = NetworkParameters(G=graph,atom_mapping=atom_mapping)
            # Define output file names based on the JSON file prefix
            prefix = filename.replace('.json', '')
            shortest_paths_file = os.path.join(results_dir,f"{prefix}_shortest_paths.txt")
            shortest_paths_with_labels_file = os.path.join(results_dir,f"{prefix}_shortest_paths_with_labels.txt")
            heatmap_file = os.path.join(results_dir, f"{prefix}_heatmap.png")
            centralities_file = os.path.join(results_dir,f"{prefix}_centralities.txt")
            edge_betweenness_file = os.path.join(results_dir,f"{prefix}_edge_betweenness.txt")
            top_nodes_file = os.path.join(results_dir,f"{prefix}_top_5_percent_nodes.txt")
            top_shortest_paths_file = os.path.join(results_dir,f"{prefix}_top_10_shortest_paths.txt")
            lengths_file = os.path.join(results_dir,f"{prefix}_shortest_path_lengths.txt")

            shortest_paths = network_parameters.compute_shortest_paths(shortest_paths_file,  # File for all shortest paths
                top_shortest_paths_file)
            centralities = network_parameters.calculate_centralities()
            network_parameters.save_centrality_measures(centralities,centralities_file)
            edge_betweenness = network_parameters.calculate_edge_betweenness()
            network_parameters.save_edge_betweenness(edge_betweenness,edge_betweenness_file)
            # Identify top 10% nodes and save them
            network_parameters.identify_top_10_percent_nodes(centralities,top_nodes_file)


def find_paths( pdb_file, results_dir, dist_cutoff_graph, source_res,target_res):
    for filename in os.listdir(results_dir):
        if filename.endswith('.json') and filename.startswith('graph_cutoff_'+ dist_cutoff_graph):
            # Construct the full path to the JSON file
            json_path = os.path.join(results_dir, filename)
            # Load the graph and atom mapping
            graph, atom_mapping = ReadFiles().load_graph_and_mapping(json_path)
            # Initialize NetworkParameters
            network_parameters = NetworkParameters(G=graph,atom_mapping=atom_mapping)
            # Define output file names based on the JSON file prefix
            visualizer = PyMOLVisualizer(pdb_file=pdb_file,atom_mapping=atom_mapping,graph=graph)
            prefix = filename.replace('.json', '')
            alt_paths_file = os.path.join(results_dir,f"{prefix}_alt_paths.txt")
            network_parameters.find_alternative_paths(source_res, target_res, alt_paths_file )
            output_pml_file = os.path.join(results_dir,f"{prefix}_alt_paths.pml")
            if os.path.exists(alt_paths_file):
                visualizer.write_pml_script_for_alternative_paths(alt_paths_file,output_pml_file)
                #print(f" 🧩  Alternative paths were being written to {alt_paths_file}")


def process_graph_files_for_communities_and_cliques(results_dir, dist_cutoff_graph, dist_cutoff_clique):
    """
    Processes graph files in the specified directory to detect communities and cliques.

    Args:
        results_dir (str): Directory containing the graph files.
        method (str): Community detection method ('leiden' or 'girvan').
    """
    start_time = time.time()  # Record the start time

    # List all .json files in the results directory
    for filename in os.listdir(results_dir):
        if filename.endswith('.json') and filename.startswith('graph_cutoff_'+ dist_cutoff_graph):
            # Construct the full path to the JSON file
            json_path = os.path.join(results_dir, filename)
            graph, atom_mapping = ReadFiles().load_graph_and_mapping(json_path)
            # Load the graph and atom mapping
            # Initialize CommunityDetector and CliqueDetector
            community_detector = CommunityDetector(G=graph, atom_mapping = atom_mapping)
            # Define output file names based on the JSON file prefix
            prefix = filename.replace('.json', '')
            # Detect communities based on selected method
            communities_leiden, modularity_leiden = community_detector.detect_communities_leiden()
            communities_file = os.path.join(results_dir,f"{prefix}_communities_leiden.txt")
            community_detector.save_communities_to_file(communities_leiden,communities_file)
            print(f" 🧩  Communities detected using Leiden algorithm saved to {communities_file}")
            
            
    for filename in os.listdir(results_dir):
        if filename.endswith('.json') and filename.startswith('graph_cutoff_'+dist_cutoff_clique):
            # Construct the full path to the JSON file
            json_path = os.path.join(results_dir, filename)
            graph, atom_mapping = ReadFiles().load_graph_and_mapping(json_path)
            # Initialize CliqueDetector
            clique_detector = CliqueDetector(G= graph, atom_mapping = atom_mapping )
            # Define output file names based on the JSON file prefix
            prefix = filename.replace('.json', '')
            cliques_file = os.path.join(results_dir, f"{prefix}_cliques.txt")
            # Detect cliques
            cliques = clique_detector.detect_cliques()
            clique_detector.save_cliques_to_file(cliques, cliques_file)



def generate_pymol_scripts(results_dir, pdb_file, dist_cutoff_graph, dist_cutoff_clique):
    """
    Generates PyMOL scripts based on the provided graph and file results.

    Args:
        results_dir (str): Directory containing result files and JSON graph files.
    """
    for filename in os.listdir(results_dir):
        if filename.endswith('.json') and filename.startswith('graph_cutoff_'+dist_cutoff_graph):
            json_path = os.path.join(results_dir, filename)
            prefix = filename.replace('.json', '')
            # Load the graph and atom mapping
            graph, atom_mapping = ReadFiles().load_graph_and_mapping(json_path)
            # Set paths for the corresponding files
            communities_file = os.path.join(results_dir,f"{prefix}_communities_leiden.txt")
            centrality_file = os.path.join(results_dir,f"{prefix}_centralities.txt")
            edge_betweenness_file = os.path.join(results_dir,f"{prefix}_edge_betweenness.txt")
            top_nodes_file = os.path.join(results_dir,f"{prefix}_top_5_percent_nodes.txt")
            paths_file = os.path.join(results_dir, f"{prefix}_paths.txt")
            output_pml_file = os.path.join(results_dir, f"{prefix}_top_5_hotspot.pml")
            output_pml_communities = os.path.join(results_dir,f"{prefix}_communities.pml")
            output_top_paths = os.path.join(results_dir,f"{prefix}_top_paths.pml")
            top_15_file = os.path.join(results_dir,f"{prefix}_top_10_shortest_paths.txt")
            pdb_output_path = os.path.join(results_dir,os.path.basename(pdb_file))
            shutil.copy(pdb_file, pdb_output_path)

            # Initialize PyMOLVisualizer
            visualizer = PyMOLVisualizer(pdb_file=pdb_file,atom_mapping=atom_mapping,graph=graph)

            # Generate PyMOL scripts
            if os.path.exists(communities_file):
                visualizer.communities_pml(communities_file,output_pml_communities)

            if os.path.exists(centrality_file) and os.path.exists(edge_betweenness_file):
                visualizer.graph_pml(centrality_file,edge_betweenness_file, os.path.join(results_dir,f"{prefix}_graph"))

            if os.path.exists(top_nodes_file):
                #print(f" ⚙️ Processing: {pdb_file}, {top_nodes_file}, {output_pml_file}")
                visualizer.highlight_top_nodes_pml(pdb_file, atom_mapping,top_nodes_file,output_pml_file)

            if os.path.exists(paths_file):
                with open(paths_file, 'r') as f:
                    residue_list = [line.strip() for line in f]
                visualizer.write_pml_script_for_residue_paths(residue_list,os.path.join(results_dir,f"{prefix}_paths.pml"))
            visualizer.write_pml_script_for_top_shortest_paths(top_15_file,edge_betweenness_file,output_top_paths)

    for filename in os.listdir(results_dir):
        if filename.endswith('.json') and filename.startswith('graph_cutoff_'+ dist_cutoff_clique):
            json_path = os.path.join(results_dir, filename)
            prefix = filename.replace('.json', '')
            cliques_file = os.path.join(results_dir,f"{prefix}_cliques.txt")
            output_pml_cliques = os.path.join(results_dir,f"{prefix}_cliques.pml")
            #print("started writing pml for cliques")
            if os.path.exists(cliques_file):
                visualizer.cliques_pml(cliques_file, output_pml_cliques)

            print(f" 🖥️  PyMOL scripts generated.")

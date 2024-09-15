# Created by sneha bheemireddy at 12/8/24
""" Run the computation of descriptors"""
import json
import numpy as np
from Bio import PDB
import networkx as nx
import sys
import time
from pprint import pprint
import os


from networkparameters import NetworkParameters  # Ensure NetworkParameters is imported
from pymol_visualizer import PyMOLVisualizer
import config as cfg
from graph_constructor import *
from read_files import *
#import functions as fs
from communities_cliques import *

# =============================================================================
# Config parsing and setup
# =============================================================================

if len(sys.argv) != 2:
    raise ValueError('\n[runner.py] syntax is: python3 runner.py path-to-config-file')

config_path = sys.argv[1]
param_space, dict_arg = cfg.parse_params(config_path) 

# Debugging output (optional)
#pprint(dict_arg)

# =============================================================================
# Graphs construction
# =============================================================================

time_gc = time.time()

def process_graphs(param_space, distance_cutoffs=None):
    """
    Constructs graphs based on distance cutoffs, processes them, and saves results.
    
    Args:
        param_space (Namespace): An object containing file paths and directories.
        distance_cutoffs (list, optional): List of distance cutoffs to use. If None, a default list is used.
    """
    if distance_cutoffs is None:
        #distance_cutoffs = [4.5, 5, 8, 10, 12]  # Default cutoffs if none are provided
        distance_cutoffs = [5]

    # Initialize GraphConstructor
    graph_constructor = GraphConstructor(
        distance_file=param_space.min_dist_matrix_file,
        adjacency_file=param_space.adjacency_file,
        distance_cutoffs=distance_cutoffs
    )

    # Iterate over each distance cutoff to build and process the graph
    for distance_cutoff in graph_constructor.distance_cutoffs:
        start_time = time.time()
        
        # Build the graph
        G = graph_constructor.build_graph_from_matrices(
            distance_file=param_space.min_dist_matrix_file,
            adjacency_file=param_space.adjacency_file,
            distance_cutoff=distance_cutoff
        )
        
        # Ensure graph connectivity
        G = graph_constructor.ensure_graph_connectivity(G)
        
        # Generate atom mapping before saving the graph
        atom_mapping, _ = graph_constructor.reader.atom_mapping(param_space.pdb_file_path)
        
        # Define file names based on the distance cutoff
        graph_filename = f"graph_cutoff_{distance_cutoff}.json"
        output_graph_file = os.path.join(param_space.results_dir, graph_filename)
        
        # Save the graph and atom mapping
        graph_constructor.save_graph_and_mapping(G, atom_mapping, output_file=output_graph_file)
        
        # Plot and save histogram
        output_file_prefix = os.path.join(param_space.results_dir, f"graph_cutoff_{distance_cutoff}")
        graph_constructor.plot_and_save_histogram(G, output_file_prefix=output_file_prefix)

        print(f"Processed graph for cutoff {distance_cutoff} in {round(time.time() - start_time, 2)} seconds")
    
    #print(param_space)
    # Write selected atoms to a PDB file using the atom mapping
    #input_pdb_path = param_space.pdb_file_path
    #output_pdb_path = os.path.join(param_space.results_dir, "atom_mapping.pdb")
    #graph_constructor.write_selected_atoms_to_pdb(input_pdb_file=input_pdb_path, output_pdb_file=output_pdb_path)

    #print(f"Atom mapping saved to {output_pdb_path}")

# Run the graph processing function
process_graphs(param_space)

print(f'\nTime constructing graphs: {round(time.time() - time_gc, 2)} s')


# =============================================================================
# Computing network parameters
# =============================================================================

# Graphs construction
time_netp = time.time()


def process_graph_files(results_dir):
    """
    Process all graph files in the specified results directory.

    Args:
        results_dir (str): Path to the directory containing the result files.
    """
    # List all .json files in the results directory
    for filename in os.listdir(results_dir):
        if filename.endswith('.json') and filename.startswith('graph_cutoff_'):
            # Construct the full path to the JSON file
            json_path = os.path.join(results_dir, filename)
            
            # Load the graph and atom mapping
            graph, atom_mapping = ReadFiles().load_graph_and_mapping(json_path)
            
            # Initialize NetworkParameters
            network_parameters = NetworkParameters(G=graph, atom_mapping=atom_mapping)
            
            # Define output file names based on the JSON file prefix
            prefix = filename.replace('.json', '')
            shortest_paths_file = os.path.join(results_dir, f"{prefix}_shortest_paths.txt")
            shortest_paths_with_labels_file = os.path.join(results_dir, f"{prefix}_shortest_paths_with_labels.txt")
            heatmap_file = os.path.join(results_dir, f"{prefix}_heatmap.png")
            centralities_file = os.path.join(results_dir, f"{prefix}_centralities.txt")
            edge_betweenness_file = os.path.join(results_dir, f"{prefix}_edge_betweenness.txt")
            top_nodes_file = os.path.join(results_dir, f"{prefix}_top_10_percent_nodes.txt")
            top_shortest_paths_file = os.path.join(results_dir, f"{prefix}_top_10_shortest_paths.txt")
            lengths_file = os.path.join(results_dir, f"{prefix}_shortest_path_lengths.txt")
            # Compute shortest paths and save them
            
            #compute_shortest_paths(self, all_paths_file, lengths_file, top_file, num_processes=32)
            # Save shortest paths with labels and create heatmap
            #network_parameters.save_paths_and_create_heatmap(
            #    shortest_path_lengths=shortest_paths,
            #    output_file=shortest_paths_with_labels_file,
            #    heatmap_file=heatmap_file,
            #    title=prefix,
            #    cbar_label="Shortest Path Length"
            
            #shortest_paths = network_parameters.compute_shortest_paths(shortest_paths_file,lengths_file,top_shortest_paths_file)
            shortest_paths = network_parameters.compute_shortest_paths(shortest_paths_file,            # File for all shortest paths
                top_shortest_paths_file,           # File for the top 10 (or 50) shortest paths
                num_processes=32)
            #network_parameters.save_paths_and_create_heatmap(shortest_paths, heatmap_file, "Shortest Paths", "Path Length")
            network_parameters.save_paths_and_create_heatmap(shortest_path_lengths=shortest_paths,       # The dictionary with shortest path lengths
            heatmap_file=os.path.join(results_dir, f"{prefix}_shortest_paths_heatmap.png"),  # Heatmap file path
            title="Shortest Paths Heatmap",cbar_label="Path Length")

            #network_parameters.write_top_shortest_paths(shortest_paths,top_shortest_paths_file)
            # Calculate and save centralities
            centralities = network_parameters.calculate_centralities()
            network_parameters.save_centrality_measures(centralities, centralities_file)
            edge_betweenness= network_parameters.calculate_edge_betweenness()
            network_parameters.save_edge_betweenness(edge_betweenness, edge_betweenness_file)
            
            # Identify top 10% nodes and save them
            network_parameters.identify_top_10_percent_nodes(centralities, top_nodes_file)
            
            # Example for finding alternative paths - replace 'source_residue' and 'target_residue' with actual values
            # Make sure you have valid source and target residues to avoid errors
            #source_residue = '269'
            #target_residue = '112'
            #alternative_paths = network_parameters.find_alternative_paths(source_residue, target_residue)
            #print(f"Alternative paths for {filename}: {alternative_paths}")

# Assuming param_space.results_dir is defined
results_dir = param_space.results_dir
process_graph_files(results_dir)

# Find alternative paths for a specific graph
#graph, atom_mapping = rf.ReadFiles().load_graph_and_mapping('path_to_graph_file.json')
#alternative_paths = network_parameters.find_alternative_paths(graph, atom_mapping, 'source_residue', 'target_residue')
print(f'Time computing network parameters:{round(time.time() - time_netp, 2)} s')


# =============================================================================
# Computing communities and cliques
# =============================================================================

time_cc = time.time()

def process_graph_files_for_communities_and_cliques(results_dir, method):
    """
    Processes graph files in the specified directory to detect communities and cliques.

    Args:
        results_dir (str): Directory containing the graph files.
        method (str): Community detection method ('leiden' or 'girvan').
    """
    start_time = time.time()  # Record the start time

    # List all .json files in the results directory
    for filename in os.listdir(results_dir):
        if filename.endswith('.json') and filename.startswith('graph_cutoff_'):
            # Construct the full path to the JSON file
            json_path = os.path.join(results_dir, filename)
            
            # Load the graph and atom mapping
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            G = nx.readwrite.json_graph.node_link_graph(data['graph'])
            
            # Initialize CommunityDetector and CliqueDetector
            community_detector = CommunityDetector(G=G)
            clique_detector = CliqueDetector(G=G)
            
            # Define output file names based on the JSON file prefix
            prefix = filename.replace('.json', '')
            cliques_file = os.path.join(results_dir, f"{prefix}_cliques.txt")
            
            # Detect communities based on selected method
            if method == 'leiden':
                communities_leiden, modularity_leiden = community_detector.detect_communities_leiden()
                communities_file = os.path.join(results_dir, f"{prefix}_communities_leiden.txt")
                community_detector.save_communities_to_file(communities_leiden, communities_file)
                print(f"Communities detected using Leiden algorithm saved to {communities_file}")
            
            elif method == 'girvan':
                communities_girvan_newman, modularity_girvan_newman = community_detector.detect_communities_girvan_newman()
                communities_file = os.path.join(results_dir, f"{prefix}_communities_girvan_newman.txt")
                community_detector.save_communities_to_file(communities_girvan_newman, communities_file)
                print(f"Communities detected using Girvan-Newman algorithm saved to {communities_file}")
            
            else:
                print(f"Invalid method '{method}' selected. Please choose 'leiden' or 'girvan'.")
                continue
            
            # Detect cliques
            cliques = clique_detector.detect_cliques()
            clique_detector.save_cliques_to_file(cliques, cliques_file)

# Prompt user for the method to use
#method = input("Select community detection method ('leiden' or 'girvan'): ").strip().lower()
method = 'leiden'
# Assuming param_space.results_dir is defined
results_dir = param_space.results_dir
process_graph_files_for_communities_and_cliques(results_dir, method)
            
print(f'Time computing communities and cliques: {round(time.time() - time_cc, 2)} s')

# =============================================================================
# Writing files for visualization
# =============================================================================

time_pml = time.time()

def generate_pymol_scripts(results_dir):
    """
    Generates PyMOL scripts based on the provided graph and file results.

    Args:
        results_dir (str): Directory containing result files and JSON graph files.
    """
    for filename in os.listdir(results_dir):
        if filename.endswith('.json') and filename.startswith('graph_cutoff_'):
            json_path = os.path.join(results_dir, filename)
            prefix = filename.replace('.json', '')
            pdb_file = param_space.pdb_file_path
            #mapped_pdb = os.path.join(results_dir, "atom_mapping.pdb")
            
            # Load the graph and atom mapping
            graph, atom_mapping = rf.ReadFiles().load_graph_and_mapping(json_path)
            #print(atom_mapping)
            
            # Set paths for the corresponding files
            communities_file = os.path.join(results_dir, f"{prefix}_communities_leiden.txt")
            cliques_file = os.path.join(results_dir, f"{prefix}_cliques_filtered.txt")
            centrality_file = os.path.join(results_dir, f"{prefix}_centralities.txt")
            edge_betweenness_file = os.path.join(results_dir, f"{prefix}_edge_betweenness.txt")
            top_nodes_file = os.path.join(results_dir, f"{prefix}_top_10_percent_nodes.txt")
            paths_file = os.path.join(results_dir, f"{prefix}_paths.txt")
            output_pml_file = os.path.join(results_dir, f"{prefix}_top_10.pml")
            output_pml_cliques = os.path.join(results_dir, f"{prefix}_cliques.pml")
            output_pml_communities = os.path.join(results_dir, f"{prefix}_communities.pml")
            output_top_paths = os.path.join(results_dir, f"{prefix}_top_paths.pml")
            top_15_file = os.path.join(results_dir, f"{prefix}_top_10_shortest_paths.txt")
            #users/sbheemir/August_2024/results/1kb4/graph_cutoff_5_top_10_shortest_paths.txt
            
            # Initialize PyMOLVisualizer
            visualizer = PyMOLVisualizer(pdb_file=pdb_file, atom_mapping=atom_mapping, graph=graph)
            
            # Generate PyMOL scripts
            if os.path.exists(communities_file):
                visualizer.communities_pml(communities_file,output_pml_communities )

            if os.path.exists(cliques_file):
                visualizer.cliques_pml(atom_mapping,cliques_file,edge_betweenness_file,output_pml_cliques)

            if os.path.exists(centrality_file) and os.path.exists(edge_betweenness_file):
                visualizer.graph_pml(atom_mapping, centrality_file, edge_betweenness_file, os.path.join(results_dir, f"{prefix}_graph"))

            if os.path.exists(top_nodes_file):
                print(f"Processing: {pdb_file}, {top_nodes_file}, {output_pml_file}")
                visualizer.highlight_top_nodes_pml(pdb_file, atom_mapping, top_nodes_file, output_pml_file)

            if os.path.exists(paths_file):
                with open(paths_file, 'r') as f:
                    residue_list = [line.strip() for line in f]
                visualizer.write_pml_script_for_residue_paths(residue_list, os.path.join(results_dir, f"{prefix}_paths.pml"))
                
            visualizer.write_pml_script_for_top_shortest_paths(top_15_file,edge_betweenness_file, output_top_paths)
            print(f"PyMOL scripts for {filename} generated.")

# Assuming param_space.results_dir is defined
results_dir = param_space.results_dir
time_pml = time.time()
generate_pymol_scripts(results_dir)
print(f'Time for generating pml files: {round(time.time() - time_pml, 2)}')


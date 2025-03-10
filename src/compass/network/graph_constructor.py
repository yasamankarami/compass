import json

import matplotlib.pyplot as plt
import networkx as nx

from compass.network import read_files as rf


class GraphConstructor:
    """
    A class to analyze molecular structures, build graphs from distance and adjacency matrices, and manage data.

    Attributes:
        distance_file (str): Path to the file containing the distance matrix.
        adjacency_file (str): Path to the file containing the adjacency matrix.
        distance_cutoffs (list): List of distance cutoffs for graph generation.
        reader (ReadFiles): Instance of ReadFiles class for reading matrices.
    """

    def __init__(self, distance_file, adjacency_file, distance_cutoffs):
        """
        Initializes the GraphConstructor with file paths and distance cutoffs.

        Args:
            distance_file (str): Path to the distance matrix file.
            adjacency_file (str): Path to the adjacency matrix file.
            distance_cutoffs (list): List of distance cutoffs for graph generation.
        """
        self.distance_file = distance_file
        self.adjacency_file = adjacency_file
        self.distance_cutoffs = distance_cutoffs
        self.reader = rf.ReadFiles()  # Create an instance of ReadFiles

    def build_graph_from_matrices(self, distance_file, adjacency_file,
                                  distance_cutoff, atom_mapping):
        """
        Builds a graph using distance and adjacency matrices with a specified distance cutoff.

        Args:
            distance_file (str): Path to the file containing the distance matrix.
            adjacency_file (str): Path to the file containing the adjacency matrix.
            distance_cutoff (float): Distance cutoff for edge inclusion.

        Returns:
            nx.Graph: The constructed graph.
        """
        # Use the ReadFiles instance to read the matrices
        min_dist_matrix = self.reader.read_matrix(distance_file)
        adjacency_matrix = self.reader.read_matrix(adjacency_file)

        G = nx.Graph()
        num_nodes = len(min_dist_matrix)

        for i in range(num_nodes):
            G.add_node(i)

        # Add edges based on the distance and adjacency matrices
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if min_dist_matrix[i, j] < int(distance_cutoff) and \
                        adjacency_matrix[i, j] > 0:
                    G.add_edge(i, j, weight=adjacency_matrix[i, j])
        return G

    def save_graph_and_mapping(self, G, atom_mapping, output_file):
        """
        Saves the graph and atom mapping to a JSON file.

        Args:
            G (nx.Graph): The graph to save.
            atom_mapping (dict): The atom mapping to save.
            output_file (str): Path to the output JSON file.
        """
        data = {
            'graph': nx.readwrite.json_graph.node_link_data(G),
            'atom_mapping': atom_mapping
        }
        with open(output_file, 'w') as f:
            json.dump(data, f)
        print(f" 📥  Graph and atom mapping saved to {output_file}")

    def plot_and_save_histogram(self, G, output_file_prefix):
        """
        Plots and saves a histogram of the graph's edge weights.

        Args:
            G (nx.Graph): The graph whose edge weights are to be plotted.
            output_file_prefix (str): Prefix for the output histogram file.
        """
        #weights = [data['weight'] for u, v, data in G.edges(data=True)]
        weights = []
        for u, v, data in G.edges(data=True):
            try:
                weights.append(data['weight'])
            except KeyError:
                weights.append(0)

        plt.figure(figsize=(10, 6))
        plt.hist(weights, bins=10, edgecolor='black', alpha=0.7)
        plt.title('Histogram of Edge Weights')
        plt.xlabel('Weight')
        plt.ylabel('Frequency')

        histogram_file = f"{output_file_prefix}_histogram.png"
        plt.savefig(histogram_file)
        plt.close()
        print(f" 📈  Histogram of edge weights saved to {histogram_file}")

    def ensure_graph_connectivity(self, G):
        """
        Ensure the graph is connected by checking and connecting components.

        Args:
            G (nx.Graph): The graph to ensure connectivity.

        Returns:
            nx.Graph: The connected graph.
        """
        if not nx.is_connected(G):
            # print("Graph is not connected. Attempting to connect components.")
            components = list(nx.connected_components(G))
            largest_component = max(components, key=len)
            subgraphs = [G.subgraph(component) for component in components]

            # Connect all components to the largest component
            for component in components:
                if component != largest_component:
                    G.add_edges_from(
                        [(list(largest_component)[0], list(component)[0])]
                    )
        return G

    def write_selected_atoms_to_pdb(self, input_pdb_file, output_pdb_file):
        """
        Write selected atoms to a new PDB file based on atom mapping.

        Args:
            input_pdb_file (str): Path to the input PDB file.
            output_pdb_file (str): Path to the output PDB file.
        """
        # Get atom mapping from the ReadFiles class
        atom_mapping, _ = self.reader.atom_mapping(input_pdb_file)

        with open(input_pdb_file, 'r') as infile, open(output_pdb_file,
                                                       'w') as outfile:
            for line in infile:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    chain_id = line[21].strip()
                    res_num = int(line[22:26].strip())
                    atom_name = line[12:16].strip()

                    # Check if this atom should be included based on atom mapping
                    for index, (
                    res_name, a_name, r_num, c_id) in atom_mapping.items():
                        if res_num == r_num and chain_id == c_id and atom_name == a_name:
                            outfile.write(line)
                            break
        print(f"Selected atoms written to {output_pdb_file}")

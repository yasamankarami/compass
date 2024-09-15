import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from multiprocessing import Pool


class NetworkParameters:
    """
    A class to compute and analyze network parameters, including shortest paths and visualization
    through heatmaps.

    Attributes:
        G (nx.Graph): The network graph object.
        atom_mapping (dict): A dictionary mapping node indices to atom information for labeling purposes.
    """

    def __init__(self, G, atom_mapping=None):
        """
        Initializes the NetworkParameters with a graph and optional atom mapping.

        Args:
            G (nx.Graph): The network graph object.
            atom_mapping (dict): A dictionary mapping node indices to atom information (optional).
        """
        self.G = G
        self.atom_mapping = atom_mapping if atom_mapping else {}

    def compute_shortest_paths(self, all_paths_file, top_file, num_processes=32):
        """
        Computes shortest paths between every pair of nodes, saves all paths with nodes involved,
        and saves the top 50 shortest paths with node details and residue mapping.

        Args:
            all_paths_file (str): Path to the file where all shortest paths with nodes will be saved.
            top_file (str): Path to the file where the top 50 shortest paths with details will be saved.
            num_processes (int): Number of processes to use for parallel computation.

        Returns:
            dict: A dictionary of shortest path lengths between nodes.
        """
        start_time = time.time()

        # Compute shortest paths and path lengths for all pairs using Dijkstra's algorithm
        shortest_paths = dict(nx.all_pairs_dijkstra_path(self.G, weight='weight'))
        shortest_path_lengths = dict(nx.all_pairs_dijkstra_path_length(self.G, weight='weight'))

        # Flatten the dictionary and sort by path length
        path_list = [
            (source, target, length)
            for source, targets in shortest_path_lengths.items()
            for target, length in targets.items()
            if source != target  # Exclude self-loops
        ]
        path_list = sorted(path_list, key=lambda x: x[2], reverse=True)

        # Write all shortest paths with nodes involved to the all_paths_file
        with open(all_paths_file, 'w') as file:
            for source, target, length in path_list:
                path = shortest_paths[source][target]
                path_str = " -> ".join(map(str, path))
                file.write(f"{source} -> {target}: Length = {length:.2f}, Path = [{path_str}]\n")

        # Write the top 50 shortest paths with residue mapping to the top_file
        self.write_top_50_shortest_paths_with_mapping(path_list[:50], shortest_paths, top_file)

        end_time = time.time()
        print(f"Shortest paths computed and saved in {end_time - start_time:.2f} seconds")
        return shortest_path_lengths

    def write_top_50_shortest_paths_with_mapping(self, top_paths, shortest_paths, top_file):
        """
        Writes the top 50 shortest paths to a file, including node details and residue mapping.

        Args:
            top_paths (list): A list of tuples representing the top shortest paths.
            shortest_paths (dict): A dictionary containing the full paths for each node pair.
            top_file (str): Path to the file where the top 50 shortest paths with details will be saved.
        """
        with open(top_file, 'w') as file:
            file.write("Top 50 Shortest Paths:\n")
            for source, target, length in top_paths:
                path = shortest_paths[source][target]
                path_str = " -> ".join(map(str, path))
                file.write(f"{source} -> {target}: Length = {length:.2f}, Path = [{path_str}]\n")
                #print(self.atom_mapping)
                
                # Map nodes to residues and write the mapped path
                mapped_path_str = " -> ".join( f"Residue {self.atom_mapping.get(str(node), ('Unknown', 'Unknown', -1, 'Unknown'))[2]} {self.atom_mapping.get(str(node), ('Unknown', 'Unknown', -1, 'Unknown'))[1]} {self.atom_mapping.get(str(node), ('Unknown', 'Unknown', -1, 'Unknown'))[3]}" for node in path )
                file.write(f"Path_mapped = [{mapped_path_str}]\n\n")

        print(f"Top 50 shortest paths with node and residue mapping written to {top_file}")

    def calculate_shortest_path_between_residues(self, residue1, residue2):
        """
        Calculates the shortest path between two residues.

        Args:
            residue1 (str): The first residue node index.
            residue2 (str): The second residue node index.

        Returns:
            tuple: The shortest path length and the path as a list of nodes.
        """
        try:
            length, path = nx.single_source_dijkstra(self.G, residue1, target=residue2)
            return length, path
        except nx.NetworkXNoPath:
            return float('inf'), []

    def save_paths_and_create_heatmap(self, shortest_path_lengths, heatmap_file, title, cbar_label):
        """
        Creates a heatmap of the shortest paths.

        Args:
            shortest_path_lengths (dict): A dictionary of shortest path lengths.
            heatmap_file (str): Path to the file where the heatmap will be saved.
            title (str): Title of the heatmap.
            cbar_label (str): Label for the color bar.
        """
        start_time = time.time()

        # Create a matrix for the heatmap
        num_nodes = len(self.G.nodes())
        data_matrix = np.zeros((num_nodes, num_nodes))
    
        for source, paths in shortest_path_lengths.items():
            for target, length in paths.items():
                data_matrix[int(source)][int(target)] = length
                data_matrix[int(target)][int(source)] = length  # Ensure symmetry

        # Replace inf values in the data matrix with 0
        data_matrix[np.isinf(data_matrix)] = 0
    
        # Determine the minimum and maximum values for the color scale
        vmin = np.min(data_matrix)
        vmax = np.max(data_matrix)
        print(vmin, vmax, "minimum and maximum values of heatmap")

        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(data_matrix, annot=False, fmt=".2f", cmap="viridis", cbar_kws={'label': cbar_label}, vmin=vmin, vmax=vmax)
        
        # Set titles and labels
        plt.title(title)
        plt.xlabel("Node Index")
        plt.ylabel("Node Index")
        
        # Save the heatmap to the specified file
        plt.savefig(heatmap_file)
        plt.close()

        end_time = time.time()
        print(f"Heatmap created and saved in {end_time - start_time:.2f} seconds")

    def find_alternative_paths(self, source_residue, target_residue, k=10):
        """
        Finds the top k alternative paths by removing edges with the least weight.

        Args:
            source_residue (str): The residue identifier for the source node.
            target_residue (str): The residue identifier for the target node.
            k (int): Number of alternative paths to find.

        Returns:
            list: A list of alternative paths.
        """
        if source_residue not in self.atom_mapping or target_residue not in self.atom_mapping:
            raise ValueError(f"One of the residues {source_residue} or {target_residue} is not in the atom_mapping.")
    
        source_node = int(source_residue)
        target_node = int(target_residue)
    
        if source_node == target_node:
            raise ValueError("Source and target residues are the same.")
    
        all_paths = list(nx.all_shortest_paths(self.G, source=source_node, target=target_node))
        if not all_paths:
            raise ValueError(f"No path found between residues {source_node} and {target_node}.")
    
        # Find all edges in all paths
        all_edges = set()
        for path in all_paths:
            all_edges.update(zip(path[:-1], path[1:]))
    
        # Remove edges with the least weight
        edges = list(all_edges)
        edges_with_weights = [(edge, self.G[edge[0]][edge[1]].get('weight', 1)) for edge in edges]
        edges_with_weights.sort(key=lambda x: x[1])  # Sort edges by weight
    
        alternative_paths = []
        for edge, _ in edges_with_weights:
            temp_graph = self.G.copy()
            temp_graph.remove_edge(edge[0], edge[1])
            try:
                new_paths = list(nx.all_shortest_paths(temp_graph, source=source_node, target=target_node))
                alternative_paths.extend(new_paths)
                if len(alternative_paths) >= k:
                    break
            except nx.NetworkXNoPath:
                continue
    
        return alternative_paths[:k]

    def calculate_centralities(self):
        """
        Calculates various centrality measures for the graph.

        Returns:
            tuple: A tuple containing dictionaries for betweenness, closeness, and degree centralities.
        """
        start_time = time.time()
        betweenness = nx.betweenness_centrality(self.G, weight='weight')
        closeness = nx.closeness_centrality(self.G, distance='weight')
        degree = dict(self.G.degree())
        end_time = time.time()
        print(f"Centralities calculated in {end_time - start_time:.2f} seconds")
        return betweenness, closeness, degree

    def save_centrality_measures(self, centralities, output_file):
        """
        Saves centrality measures to a file.

        Args:
            centralities (tuple): A tuple containing dictionaries for betweenness, closeness, and degree centralities.
            output_file (str): Path to the file where centrality measures will be saved.
        """
        start_time = time.time()
        betweenness, closeness, degree = centralities
        with open(output_file, 'w') as f:
            f.write("Node\tBetweenness\tCloseness\tDegree\n")
            for node in betweenness.keys():
                f.write(f"{node}\t{betweenness[node]:.4f}\t{closeness[node]:.4f}\t{degree[node]}\n")
        end_time = time.time()
        print(f"Centralities saved in {end_time - start_time:.2f} seconds")
        
    def calculate_edge_betweenness(self):
        """
        Calculates edge betweenness centrality for the graph.

        Returns:
            dict: A dictionary mapping edges to their betweenness centrality value.
        """
        start_time = time.time()
        edge_betweenness = nx.edge_betweenness_centrality(self.G, weight='weight')
        end_time = time.time()
        print(f"Edge betweenness calculated in {end_time - start_time:.2f} seconds")
        return edge_betweenness

    def save_edge_betweenness(self, edge_betweenness, output_file):
        """
        Saves edge betweenness centrality measures to a file.

        Args:
            edge_betweenness (dict): A dictionary mapping edges to their betweenness centrality value.
            output_file (str): Path to the file where edge betweenness centralities will be saved.
        """
        start_time = time.time()
        with open(output_file, 'w') as f:
            f.write("Edge\tBetweenness\n")
            for edge, centrality in edge_betweenness.items():
                # Edge format: (node1, node2)
                edge_str = f"{edge[0]}-{edge[1]}"
                f.write(f"{edge_str}\t{centrality:.4f}\n")
        end_time = time.time()
        print(f"Edge betweenness saved in {end_time - start_time:.2f} seconds")

    def identify_top_10_percent_nodes(self, centralities, output_file):
        """
        Identifies the top 5% nodes based on centrality measures and saves them as allosteric hotspots.

        Args:
            centralities (tuple): A tuple containing dictionaries for betweenness, closeness, and degree centralities.
            output_file (str): Path to the file where top nodes will be saved.
        """
        betweenness, closeness, degree = centralities
        num_nodes = len(betweenness)
        top_n = max(1, num_nodes // 20)

        # Sorting nodes based on centrality measures
        sorted_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:top_n]
        sorted_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:top_n]
        sorted_degree = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # Merging all top nodes
        top_nodes = set([node for node, _ in sorted_betweenness] +
                        [node for node, _ in sorted_closeness] +
                        [node for node, _ in sorted_degree])

        with open(output_file, 'w') as f:
            f.write("Top 10% Nodes (Allosteric Hotspots):\n")
            for node in top_nodes:
                f.write(f"Node {node}\n")

        print(f"Top 10% nodes identified and saved as allosteric hotspots ")


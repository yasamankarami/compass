import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
import networkx as nx
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
from itertools import combinations
from dataclasses import dataclass, field
from typing import Any
import heapq


class NetworkParameters:
    """
    A class to compute and analyze network parameters, focusing on shortest paths
    and their visualization through path mapping.

    This implementation uses an efficient sequential algorithm to compute shortest
    paths between nodes in the network, with special handling for atom/residue mapping
    in molecular networks.

    Attributes:
        G (nx.Graph): The network graph object.
        atom_mapping (dict): Maps node indices to atom information (chain, residue name, 
                           residue number, atom name).
    """
    
    def __init__(self, G, atom_mapping=None):
        """
        Initialize the NetworkParameters class.

        Args:
            G (nx.Graph): Network graph with weighted edges.
            atom_mapping (dict, optional): Dictionary mapping node indices to atom information.
                Expected format: {node_id: (chain, residue_name, residue_number, atom_name)}
        """
        self.G = G
        self.atom_mapping = atom_mapping if atom_mapping else {}

    def compute_shortest_paths(self, all_paths_file, top_file):
        """
        Compute shortest paths between nodes until reaching 20% of total residues.
        
        The method uses an optimized sequential approach that:
        1. Computes shortest paths between all node pairs
        2. Sorts paths by length
        3. Collects paths until reaching the residue threshold
        4. Saves results to specified files
        
        Args:
            all_paths_file (str): Path to save all computed shortest paths
            top_file (str): Path to save detailed top paths with residue mapping

        Returns:
            dict: Dictionary of path lengths between node pairs
        """
        #print("Starting shortest path computations...")
        nodes = sorted(list(self.G.nodes()))  # Sort nodes for consistent results
        
        # Compute all shortest paths
        shortest_paths, path_lengths = self._compute_all_shortest_paths(nodes)
        
        # Process paths and apply residue threshold
        collected_paths = self._collect_paths_until_threshold(
            nodes, shortest_paths, path_lengths
        )
        #print(len(shortest_paths),"shortest_paths", print(len(collected_paths),"collected paths"))
        # Save results to files
        self._save_paths(all_paths_file, top_file, collected_paths, shortest_paths)
        
        return path_lengths

    def _compute_all_shortest_paths(self, nodes):
        """
        Compute shortest paths between all node pairs efficiently.
        
        Uses single_source_dijkstra to compute paths from each source node
        to all possible targets in one pass, improving performance.
        
        Args:
            nodes (list): Sorted list of node indices
            
        Returns:
            tuple: (shortest_paths, path_lengths) dictionaries
        """
        start_time = time.time()
        shortest_paths = {}
        path_lengths = {}
        
        for source in nodes:
            try:
                # Compute all shortest paths from source in one call
                distances, paths = nx.single_source_dijkstra(self.G, source, weight='weight')
                
                # Store only paths to nodes that come after source in sorted order
                # This avoids redundant path storage
                for target in (n for n in nodes if n > source):
                    if target in paths:
                        shortest_paths[(source, target)] = paths[target]
                        path_lengths[(source, target)] = distances[target]
            except nx.NetworkXNoPath:
                continue
        
        end_time = time.time()
        print(f" 📐  Shortest paths computation completed in {end_time - start_time:.2f} seconds")
        return shortest_paths, path_lengths

    def _collect_paths_until_threshold(self, nodes, shortest_paths, path_lengths):
        """
        Collect paths until reaching the residue threshold (20% of total residues).
        
        Args:
            nodes (list): List of all nodes
            shortest_paths (dict): Dictionary of shortest paths
            path_lengths (dict): Dictionary of path lengths
            
        Returns:
            list: Collected paths that meet the threshold criterion
        """
        total_residues = len(nodes)
        residue_threshold = 0.2 * total_residues
        
        # Create and sort path list by length (descending)
        path_list = [
            (source, target, length)
            for (source, target), length in path_lengths.items()
        ]
        path_list.sort(key=lambda x: x[2], reverse=True)
        
        # Collect paths until reaching threshold
        collected_paths = []
        unique_residues = set()
        
        for source, target, length in path_list:
            path = shortest_paths.get((source, target))
            if path is None:
                continue
                
            collected_paths.append((source, target, length))
            unique_residues.update(path)
            
            if len(unique_residues) >= residue_threshold:
                #print(f"\nReached {len(unique_residues)} residues "
                #      f"({(len(unique_residues)/total_residues)*100:.2f}% of total)")
                break
                
        return collected_paths

    def _save_paths(self, all_paths_file, top_file, collected_paths, shortest_paths):
        """
        Save computed paths to output files.
        
        Args:
            all_paths_file (str): File to save all collected paths
            top_file (str): File to save detailed top paths with residue mapping
            collected_paths (list): List of collected path information
            shortest_paths (dict): Dictionary of shortest paths
        """
        # Write all collected paths
        with open(all_paths_file, 'w') as file:
            for source, target in shortest_paths:
                path = shortest_paths.get((source, target))
                if path:
                    path_str = " -> ".join(map(str, path))
                    file.write(f"{source} -> {target}: Path = [{path_str}]\n")
                    #res_name, atom_name, res_num, chain_id = self.atom_mapping[str(node)]
                    file.write(f"\n")
                    for node in path:
                        res_name, atom_name, res_num, chain_id = self.atom_mapping.get(str(node), ("Unknown", "Unknown", "Unknown", "Unknown"))
                        file.write(f"{res_num}:{chain_id} -> ")
                    file.write(f"\n")
        
        # Write detailed top paths with residue mapping
        self.write_top_50_shortest_paths_with_mapping(collected_paths, shortest_paths, top_file)

    def write_top_50_shortest_paths_with_mapping(self, top_paths, shortest_paths, top_file):
        """
        Write detailed path information including residue mapping.
        
        Args:
            top_paths (list): List of (source, target, length) tuples
            shortest_paths (dict): Dictionary of shortest paths
            top_file (str): Output file path
        """
        with open(top_file, 'w') as file:
            file.write("Top 50 Shortest Paths:\n")
            unique_top_paths = set()
            
            for source, target, length in top_paths:
                # Normalize path order to avoid duplicates
                if source > target:
                    source, target = target, source

                if (source, target) not in unique_top_paths:
                    try:
                        path = shortest_paths.get((source, target))
                        if not path:
                            continue
                            
                        # Write path with node indices
                        path_str = " -> ".join(map(str, path))
                        file.write(f"{source} -> {target}: Length = {length:.2f}, Path = [{path_str}]\n")
                        #file.write(f"{source} -> {target}: Length = {length:.2f}", f"Path = [{path_str}]\n")
                        '''

                        # Write path with residue mapping
                        mapped_path = []
                        for node in path:
                            atom_info = self.atom_mapping.get(str(node), ('Unknown', 'Unknown', -1, 'Unknown'))
                            mapped_path.append(f"Residue {atom_info[2]} {atom_info[1]} {atom_info[3]}")
                        
                        mapped_path_str = " -> ".join(mapped_path)
                        file.write(f"Path_mapped = [{mapped_path_str}]\n\n")
                        unique_top_paths.add((source, target))
                        '''
                    except Exception as e:
                        print(f"Error processing path {source} -> {target}: {str(e)}")
                        

        print(f" 📥  Top 50 shortest paths with node and residue mapping written to {top_file}")
        
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
        #print(vmin, vmax, "minimum and maximum values of heatmap")

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


    def find_alternative_paths(self, source_residue, target_residue, alt_paths_file, k=10):
        """
        Finds the top k alternative paths using Yen's algorithm.

        Args:
            source_residue (str): The residue identifier for the source node.
            target_residue (str): The residue identifier for the target node.
            alt_paths_file (str): File to store alternative paths.
            k (int): Number of alternative paths to find.

        Returns:
            list: A list of alternative paths.
        """
        source_res_num, source_chain_id = source_residue.split(":")
        target_res_num, target_chain_id = target_residue.split(":")

        node1 = None
        node2 = None
        for key, values in self.atom_mapping.items():
            if values[-2] == int(source_res_num) and values[-1] == source_chain_id:
                node1 = key
            if values[-2] == int(target_res_num) and values[-1] == target_chain_id:
                node2 = key
            if node1 is not None and node2 is not None:
                break

        node1, node2 = int(node1), int(node2)

        def find_yen_k_paths():
            """Implementation of Yen's k shortest paths algorithm."""
            A = []  # List of found paths
            B = []  # Candidate paths heap

            # Find the initial shortest path
            try:
                path = nx.shortest_path(self.G, node1, node2, weight='weight')
                A.append(path)
            except nx.NetworkXNoPath:
                return []

            # Find k-1 more paths
            for _ in range(1, k):
                prev_path = A[-1]

                # Examine each node in the previous path
                for i in range(len(prev_path)-1):
                    spur_node = prev_path[i]
                    root_path = prev_path[:i+1]

                    # Store removed edges to restore later
                    removed_edges = []

                    # Remove edges that were part of previous paths
                    for path in A:
                        if len(path) > i and path[:i+1] == root_path:
                            u, v = path[i], path[i+1]
                            if self.G.has_edge(u, v):
                                removed_edges.append((u, v, self.G[u][v].copy()))
                                self.G.remove_edge(u, v)

                    try:
                        # Find new path from spur node to target
                        spur_path = nx.shortest_path(self.G, spur_node, node2, weight='weight')
                        total_path = root_path[:-1] + spur_path
                        if total_path not in B:
                            heapq.heappush(B, total_path)
                    except nx.NetworkXNoPath:
                        pass

                    # Restore removed edges
                    for u, v, data in removed_edges:
                        self.G.add_edge(u, v, **data)

                if not B:
                    break

                # Add the best candidate to solution
                new_path = heapq.heappop(B)
                A.append(new_path)

            return A

        # Find paths using Yen's algorithm
        paths = find_yen_k_paths()

        # Write paths to the specified file
        with open(alt_paths_file, 'w') as f:
            f.write(f"Shortest Path: {' -> '.join(map(str, paths[0]))} \n")
            for i, path in enumerate(paths[1:k+1], start=1):
                path_str = " -> ".join(map(str, path))
                f.write(f"Alternative Path {i}: {path_str}\n")

        return paths[:k+1]  # Return the shortest path and top k alternatives

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
        #print(f"Centralities calculated in {end_time - start_time:.2f} seconds")
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
        #print(f"Centralities saved in {end_time - start_time:.2f} seconds")
        
    def calculate_edge_betweenness(self):
        """
        Calculates edge betweenness centrality for the graph.

        Returns:
            dict: A dictionary mapping edges to their betweenness centrality value.
        """
        start_time = time.time()
        #print(f"Number of nodes: {self.G.number_of_nodes()}")
        #print(f"Number of edges: {self.G.number_of_edges()}")
        # Print edges in the graph to cross-check
        #print(f"Edges in the graph: {list(self.G.edges(data=True))}")
        #print(self.G)
        edge_betweenness = nx.edge_betweenness_centrality(self.G, weight='weight')
        #print(self.G.edges())
        #print(np.max(edge_betweenness), "max edge betweenness")
        end_time = time.time()
        #print(f"Edge betweenness calculated in {end_time - start_time:.2f} seconds")
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
                #print(edge, centrality)
                if self.G.has_edge(*edge):  # Ensure that only edges in the graph are saved
                    edge_str = f"{edge[0]}-{edge[1]}"
                    f.write(f"{edge_str}\t{centrality:.4f}\n")
                # Edge format: (node1, node2)
                #edge_str = f"{edge[0]}-{edge[1]}"
                #f.write(f"{edge_str}\t{centrality:.4f}\n")
        end_time = time.time()
        #print(f"Edge betweenness saved in {end_time - start_time:.2f} seconds")

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
            f.write("Top 5% Nodes (Allosteric Hotspots):\n")
            for node in top_nodes:
                f.write(f"Node {node}\n")

        print(f" 📥  Top 5% nodes identified and saved as allosteric hotspots to {output_file} ")

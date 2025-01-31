from typing import Dict, List

import igraph as ig
import leidenalg as la
import networkx as nx


class CommunityDetector:
    """
    A class for detecting communities in a graph using various algorithms.

    Attributes:
        G (nx.Graph): The input graph for community detection.
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

    def detect_communities_leiden(self):
        """
        Detects communities using the Leiden algorithm.

        Returns:
            tuple: A tuple containing:
                - communities (dict): A dictionary mapping nodes to their community index.
                - modularity (float): The modularity of the partition.
        """
        # Convert NetworkX graph to igraph graph
        edges = list(self.G.edges())
        nodes = list(self.G.nodes())

        # Create igraph with correct node mapping
        g = ig.Graph()
        g.add_vertices(len(nodes))
        # Create node name mapping
        node_mapping = {node: idx for idx, node in enumerate(nodes)}
        # Add edges using the mapping
        edge_list = [(node_mapping[edge[0]], node_mapping[edge[1]]) for edge in
                     edges]
        g.add_edges(edge_list)

        # Run Leiden algorithm
        partition = la.find_partition(g, la.ModularityVertexPartition)
        modularity = partition.modularity

        # Convert partition to node-community mapping
        communities = {}
        for idx, community in enumerate(partition):
            for node_idx in community:
                original_node = nodes[node_idx]
                communities[original_node] = idx

        print(
            f" 🧩  Leiden detected {len(set(communities.values()))} communities with modularity {modularity:.4f}.")

        return communities, modularity

    @staticmethod
    def _collect_members_partition(self, communities):
        """
        Organizes communities by community index.

        Args:
            communities (dict): Dictionary mapping nodes to their community index.

        Returns:
            dict: Dictionary mapping community indices to lists of nodes.
        """
        community_groups = {}
        for node, comm_idx in communities.items():
            if comm_idx not in community_groups:
                community_groups[comm_idx] = []
            # print(self.atom_mapping.keys())
            res_name, atom_name, res_num, chain_id = self.atom_mapping.get(
                str(node), ("Unknown", "Unknown", "Unknown", "Unknown"))
            str_node_details = f"{chain_id}_{res_num}"
            community_groups[comm_idx].append(str_node_details)
        return community_groups

    def save_communities_to_file(self, communities, output_file):
        """
        Saves communities to a file.

        Args:
            communities (dict): A dictionary mapping nodes to their community index.
            output_file (str): Path to the output file.
        """
        try:
            # First, organize communities by community index
            community_groups = self._collect_members_partition(self,
                                                               communities)
            with open(output_file, 'w') as f:
                f.write(
                    f"Communities file. This file contains the community members information in the order chain_id residue_number\n")
                for comm_idx, members in sorted(community_groups.items()):
                    members_str = ', '.join(map(str, members))
                    f.write(f"Community {comm_idx}: {members_str}\n")
            print(f" 🧩  Communities saved to {output_file}")
        except Exception as e:
            print(f"Error writing communities to {output_file}: {e}")
            raise


class CliqueDetector:
    """
    A class for detecting cliques in a graph and saving them to files.

    Attributes:
        G (nx.Graph): The input graph for clique detection.
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

    def detect_cliques(self) -> Dict[int, List]:
        """
        Detects cliques in the graph and returns a dictionary of cliques.

        Returns:
            dict: A dictionary where keys are clique indices and values are lists of nodes in each clique.
        """
        try:
            # all_cliques = list(nx.find_cliques(G))
            all_cliques = list(nx.find_cliques(self.G))
            large_cliques = [clique for clique in all_cliques if
                             len(clique) > 2]
            sorted_large_cliques = sorted(large_cliques, key=len, reverse=True)
            selected_cliques = []
            used_nodes = set()

            # Greedily select cliques that do not overlap with previously selected ones
            for clique in sorted_large_cliques:
                # Check if clique has no overlap with used nodes
                if not any(node in used_nodes for node in clique):
                    selected_cliques.append(clique)
                    used_nodes.update(clique)  # Mark nodes as used

            clique_dict = self._collect_members_cliques(selected_cliques)

            return clique_dict
        except Exception as e:
            print(f"Error detecting cliques: {e}")
            raise

    @staticmethod
    def _collect_members_cliques(cliques: List[List]) -> Dict[int, List]:
        """
        Collects cliques into a dictionary format.

        Args:
            cliques (list): List of cliques detected by NetworkX.

        Returns:
            dict: Dictionary mapping clique indices to lists of nodes in each clique.
        """
        return {idx: sorted(clique) for idx, clique in enumerate(cliques)}

    def save_cliques_to_file(self, cliques: Dict[int, List],
                             output_file: str) -> None:
        """
        Saves all cliques to a file and filtered cliques (with more than two members)
        to another file.

        Args:
            cliques (dict): A dictionary where keys are clique indices and values are
                           lists of nodes in each clique.
            output_file (str): Path to the file where all cliques will be saved.

        Raises:
            IOError: If there's an error writing to the files.
        """

        def get_details(member):
            # Fetch atom details for each member
            res_name, atom_name, res_num, chain_id = self.atom_mapping.get(
                str(member), ("Unknown", "Unknown", "Unknown", "Unknown"))
            # Prepare a string representing the node's details
            str_node_details = f"{chain_id}_{res_num}"
            return str_node_details

        try:
            with open(output_file, 'w') as all_cliques_file:
                all_cliques_file.write(
                    f"Cliques file. This file contains the clique members information in the order chain_id residue_number\n")
                for clique_idx, members in cliques.items():
                    # Apply get_details to each member in the clique
                    new_members = [get_details(member) for member in members]
                    # Write the clique with its members to the file
                    all_cliques_file.write(
                        f"Clique {clique_idx}: {', '.join(new_members)}\n")

            print(f"🧩  Cliques saved to {output_file}")

        except IOError as e:
            print(f"Error writing cliques to file: {e}")
            raise

import igraph as ig
import leidenalg as la
import networkx as nx
from networkx.algorithms.community import girvan_newman

class CommunityDetector:
    """
    A class for detecting communities in a graph using various algorithms.

    Attributes:
        G (nx.Graph): The input graph for community detection.
    """

    def __init__(self, G):
        """
        Initializes the CommunityDetector with a graph.

        Args:
            G (nx.Graph): The input graph for community detection.
        """
        self.G = G

    def detect_communities_leiden(self):
        """
        Detects communities using the Leiden algorithm.

        Returns:
            tuple: A tuple containing:
                - communities (dict): A dictionary mapping nodes to their community index.
                - modularity (float): The modularity of the partition.
        """
        # Convert NetworkX graph to igraph graph
        g = ig.Graph.TupleList(self.G.edges(), directed=False)
        partition = la.find_partition(g, la.ModularityVertexPartition)
        modularity = partition.modularity
        communities = {}
        for idx, community in enumerate(partition):
            for node in community:
                communities[node] = idx
        
        # Collect community members
        community_members = {}
        for node, community in communities.items():
            if community not in community_members:
                community_members[community] = []
            community_members[community].append(node)
        
        print(f"Leiden detected {len(community_members)} communities with modularity {modularity:.4f}.")
        #for community, members in community_members.items():
        #    print(f"Community {community}: {members}")
        
        return communities, modularity

    def detect_communities_girvan_newman(self):
        """
        Detects communities using the Girvan-Newman algorithm.

        Returns:
            tuple: A tuple containing:
                - communities (dict): A dictionary mapping nodes to their community index.
                - modularity (float): The modularity of the partition.
        """
        comp = girvan_newman(self.G)
        limited = tuple(sorted(c) for c in next(comp))
        communities = {}
        for idx, community in enumerate(limited):
            for node in community:
                communities[node] = idx

        # Collect community members
        community_members = {}
        for node, community in communities.items():
            if community not in community_members:
                community_members[community] = []
            community_members[community].append(node)
        
        # Calculate modularity
        modularity = nx.algorithms.community.quality.modularity(self.G, limited)
        
        print(f"Girvan-Newman detected {len(community_members)} communities with modularity {modularity:.4f}.")
        #for community, members in community_members.items():
        #    print(f"Community {community}: {members}")
        
        return communities, modularity

    def save_communities_to_file(self, communities, output_file):
        """
        Saves communities to a file.

        Args:
            communities (dict): A dictionary where keys are community indices and values are lists or single nodes in each community.
            output_file (str): Path to the output file.
        """
        try:
            with open(output_file, 'w') as f:
                for community, members in communities.items():
                    # Check if members is a list or a single item
                    if isinstance(members, list):
                        members_str = ', '.join(map(str, members))
                    else:
                        # If members is a single item (int), wrap it in a list
                        members_str = str(members)
                    
                    # Write each community index and its members to the file
                    f.write(f"Community {members_str}: {community}\n")
            print(f"Communities saved to {output_file}")
        except Exception as e:
            print(f"Error writing communities to {output_file}: {e}")


class CliqueDetector:
    """
    A class for detecting cliques in a graph and saving them to files.

    Attributes:
        G (nx.Graph): The input graph for clique detection.
    """

    def __init__(self, G):
        """
        Initializes the CliqueDetector with a graph.

        Args:
            G (nx.Graph): The input graph for clique detection.
        """
        self.G = G

    def detect_cliques(self):
        """
        Detects cliques in the graph and returns a dictionary of cliques.

        Returns:
            dict: A dictionary where keys are clique indices and values are lists of nodes in each clique.
        """
        cliques = list(nx.find_cliques(self.G))
        
        # Create a dictionary to store the cliques
        clique_dict = {}
        for idx, clique in enumerate(cliques):
            for node in clique:
                clique_dict[node] = idx
        
        # Collect clique members
        clique_members = {}
        for node, clique in clique_dict.items():
            if clique not in clique_members:
                clique_members[clique] = []
            clique_members[clique].append(node)
        
        print(f"Detected {len(clique_members)} cliques.")
        return clique_members

    def save_cliques_to_file(self, cliques, output_file):
        """
        Saves all cliques to a file and filtered cliques (with more than two members) to another file.

        Args:
            cliques (dict): A dictionary where keys are clique indices and values are lists of nodes in each clique.
            output_file (str): Path to the file where all cliques will be saved.
        """
        # Generate the filtered output file name by appending "_filtered.txt" to the base file name
        filtered_output_file = output_file.replace(".txt", "_filtered.txt")
        
        with open(output_file, 'w') as all_cliques_file, open(filtered_output_file, 'w') as filtered_cliques_file:
            for clique, members in cliques.items():
                # Save all cliques
                all_cliques_file.write(f"Clique {clique}: {', '.join(map(str, members))}\n")
                
                # Save only cliques with more than two members
                if len(members) > 2:
                    filtered_cliques_file.write(f"Clique {clique}: {', '.join(map(str, members))}\n")

        print(f"All cliques saved to {output_file}")
        print(f"Filtered cliques saved to {filtered_output_file}")


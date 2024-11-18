import random
from Bio import PDB
import re
import re
from collections import defaultdict
from compass.network.read_files import ReadFiles

class PyMOLVisualizer:
    """
    A class for generating PyMOL scripts to visualize molecular structures and graph properties.

    Attributes:
        pdb_file (str): Path to the input PDB file.
        atom_mapping (dict): A dictionary mapping residue indices to their atom information.
        graph (nx.Graph): The graph object containing nodes and edges.
    """

    def __init__(self, pdb_file, atom_mapping, graph):
        """
        Initializes the PyMOLVisualizer with the PDB file, atom mapping, and graph.

        Args:
            pdb_file (str): Path to the input PDB file.
            atom_mapping (dict): A dictionary mapping residue indices to their atom information.
        """
        self.pdb_file = pdb_file
        self.atom_mapping = atom_mapping
        self.graph = graph

        #print(self.atom_mapping)

    def parse_communities_file(self, communities_file):
        """
        Parses the communities file and returns a dictionary of communities.

        Args:
            communities_file (str): Path to the file containing communities.

        Returns:
            dict: A dictionary where keys are community indices and values are lists of nodes.
        """
        communities = {}
        with open(communities_file, 'r') as f:
            for line in f:
                parts = line.strip().split(':')
                if len(parts) == 2:
                    #print('\n',"parts 0",parts[0].split()[1], "parts 1", parts[1])
                    community_idx = parts[0].split()[1]
                    node_list = parts[1].strip()
                    #nodes = [int(node.strip()) for node in node_list.split(',')]
                    if community_idx not in communities:
                        communities[community_idx] = []

                    communities[community_idx].append(node_list)

        return communities

    def communities_pml(self, communities_file, output_pml_file):
        """
        Generates a PyMOL script to visualize communities.

        Args:
            communities_file (str): Path to the file containing communities.
            output_pml_file (str): Path to the output PyMOL script file.
        """
        # Parse the communities file
        communities = self.parse_communities_file(communities_file)

        # Initialize the PDB parser and load the structure
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('structure', self.pdb_file)

        # List of standard colors (RGB format) for the top 10 communities
        standard_colors = [
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
            [0.0, 1.0, 1.0],  # Cyan
            [1.0, 0.0, 1.0],  # Magenta
            [0.5, 0.5, 0.5],  # Gray
            [1.0, 0.5, 0.0],  # Orange
            [0.5, 0.0, 0.5],  # Purple
            [0.5, 0.5, 0.0],  # Olive
        ]

        def generate_random_color():
            """Generate a random color in RGB format."""
            return [random.random(), random.random(), random.random()]

        # Assign colors to communities
        community_colors = {}
        for i, community in enumerate(communities):
            if i < len(standard_colors):
                community_colors[community] = standard_colors[i]
            else:
                community_colors[community] = generate_random_color()

        with open(output_pml_file, 'w') as f:
            # Write the command to load the structure
            f.write(f"load {self.pdb_file}, structure\n")

            # Define colors for each community
            for community, color in community_colors.items():
                f.write(f"set_color color{community}, [{', '.join(map(str, color))}]\n")

            # Apply colors to residues based on community assignment
            for community, nodes in communities.items():
                #print(community, type(community), nodes, type(nodes), len(nodes))
                color = f"color{community}"
                nodes = [int(num) for num in nodes[0].split(',')]
                for node in nodes:
                    #print(node)
                    #print(type(node),node)
                    #node_list = [int(num.strip()) for num in node.split(',')]
                    #print(type(self.atom_mapping))
                    mapped_res = self.atom_mapping[str(node)]
                    #print(f"color {color}, chain {mapped_res[3]} and resi {mapped_res[2]}\n")
                    f.write(f"color {color}, chain {mapped_res[3]} and resi {mapped_res[2]}\n")
            # Show the structure as cartoon

            f.write("show cartoon\n")
            f.write("bg_color white\n")

        print(f" 🧊  PyMOL script for communities saved to {output_pml_file}")

    def graph_pml(self, centrality_file, edge_betweenness_file, output_pml):
        """
        Generates PyMOL scripts to visualize the graph with centrality and edge betweenness.

        Args:
            atom_mapping (str): Path to the PDB atom mapping file.
            centrality_file (str): Path to the file containing centrality values.
            edge_betweenness_file (str): Path to the file containing edge betweenness values.
            output_pml (str): Prefix for the output PyMOL script files.
        """

        # Read centrality and edge betweenness from files
        centrality = ReadFiles.read_centrality_from_file(centrality_file)
        if not centrality:
            print("Warning: No centrality values found in the centrality file.")

        edge_betweenness = ReadFiles.read_edge_betweenness_from_file(edge_betweenness_file)
        if not edge_betweenness:
            print("Warning: No edge betweenness values found in the edge betweenness file.")
        #print(len(centrality), "centrality", centrality)
         # Ensure the number of nodes in the graph matches the number of residues
        #if len(centrality_file) != len(residues):
        #    print(f"Error: The number of nodes in the graph does not match the number of residues. Nodes: {len(centrality_file)}, Residues: {len(residues)}.")
        #    return


        output_files = {
            "all": open(f"{output_pml}_all.pml", 'w'),
            "ca_ca": open(f"{output_pml}_ca_ca.pml", 'w'),
            "ca_other": open(f"{output_pml}_ca_other.pml", 'w'),
            "other_other": open(f"{output_pml}_other_other.pml", 'w')
        }

        for file in output_files.values():
            file.write(f"load {self.pdb_file}, structure\n")
            file.write(f"show_as cartoon, structure\n")
            file.write(f"set cartoon_transparency,0.6\n")

        # Set sphere size based on centrality
        max_centrality = max(centrality.values(), default=1)  # Avoid division by zero
        for node, centrality_value in centrality.items():
            # Use the node number to index directly into the residues list
            res_name, atom_name, res_num, chain_id = self.atom_mapping[str(node)]
            #if node < len(residues):
            #    chain_id, res_num, atom_name = residues[node]
            norm_centrality = centrality_value / max_centrality
            sphere_scale = 1 * norm_centrality  # Simplified sphere scale calculation

            for file in output_files.values():
                file.write(f"show spheres, chain {chain_id} and resi {res_num} and name {atom_name}\n")
                file.write(f"set sphere_scale, {sphere_scale:.2f}, chain {chain_id} and resi {res_num} and name {atom_name}\n")


        # Draw edges with thickness based on edge betweenness
        max_betweenness = max(edge_betweenness.values(), default=1)  # Avoid division by zero
        for (node1, node2), betweenness_value in edge_betweenness.items():
            res_name1, atom_name1, res_num1, chain_id1 = self.atom_mapping[str(node1)]
            res_name2, atom_name2, res_num2, chain_id2 = self.atom_mapping[str(node2)]
            
            norm_betweenness = betweenness_value / max_betweenness
            thickness = 1 + 10 * norm_betweenness

            if atom_name1 == 'CA' and atom_name2 == 'CA':
                output_file = output_files["ca_ca"]
            elif atom_name1 == 'CA' or atom_name2 == 'CA':
                output_file = output_files["ca_other"]
            else:
                output_file = output_files["other_other"]

            output_file.write(f"distance edge_{node1}_{node2}, chain {chain_id1} and resi {res_num1} and name {atom_name1}, chain {chain_id2} and resi {res_num2} and name {atom_name2}\n")
            output_file.write(f"set dash_width, {thickness:.2f}, edge_{node1}_{node2}\n")
            output_files["all"].write(f"distance edge_{node1}_{node2}, chain {chain_id1} and resi {res_num1} and name {atom_name1}, chain {chain_id2} and resi {res_num2} and name {atom_name2}\n")
            output_files["all"].write(f"set dash_width, {thickness:.2f}, edge_{node1}_{node2}\n")
            
        for file in output_files.values():
            file.write("hide labels \n")
            file.write("set dash_gap,0 \n")
            file.write("set dash_color, grey10 \n")
            file.write("bg_color white\n")
            file.close()

        print(f" 🧊  PyMOL script for graph attributes saved with prefix {output_pml}")

    def parse_cliques_file(self, cliques_file):
        """
        Parses the cliques file and returns a dictionary of cliques.

        Args:
            cliques_file (str): Path to the file containing cliques.

        Returns:
            dict: A dictionary where keys are clique indices and values are lists of nodes.
        """
        cliques = {}
        with open(cliques_file, 'r') as f:
            for line in f:
                parts = line.strip().split(':')
                if len(parts) == 2:
                    clique_idx = int(parts[0].split()[1])
                    nodes = list(map(int, parts[1].split(',')))
                    cliques[clique_idx] = nodes
        return cliques

    def cliques_pml(self, cliques_file, output_pml):
        """
        Generates a PyMOL script to visualize cliques.

        Args:
            atom_mapping (str): Path to the PDB atom mapping file.
            cliques_file (str): A file mapping nodes to their clique index.
            edge_betweenness_file (str): Path to the file containing edge betweenness values.
            output_pml (str): Path to the output PyMOL script file.
        """
        # Parse the cliques file
        cliques = self.parse_cliques_file(cliques_file)
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('structure', self.pdb_file)

        # Assign a unique color to each clique and generate PyMOL commands
        with open(output_pml, 'w') as f:
            f.write(f"load {self.pdb_file}, structure\n")
            f.write("show cartoon\n")
            f.write("set cartoon_color, grey90\n")

            for i, (clique, nodes) in enumerate(cliques.items(), start=1):
                color = [random.random(), random.random(), random.random()]
                f.write(f"set_color clique_{i}, [{', '.join(map(str, color))}]\n")

                # Create a selection for this clique
                selection_residues = []
                for node in nodes:
                    res_name, atom_name, res_num, chain_id = self.atom_mapping[str(node)]
                    f.write(f"color clique_{i}, chain {chain_id} and resi {res_num}\n")
                    f.write(f"show spheres, chain {chain_id} and resi {res_num} and name {atom_name}\n")

                # Combine all residue selections into one command
                if selection_residues:
                    f.write(f"select clique_{i}, {' + '.join(selection_residues)}\n")
            f.write("set dash_gap, 0\n")
            f.write("set dash_color, grey10\n")
            f.write("hide labels\n")
            f.write("set sphere_scale, 0.7\n")
            f.write("bg_color white\n")

        print(f" 🧊  PyMOL script for cliques saved to {output_pml}")



    def highlight_top_nodes_pml(self, pdb_file, atom_mapping, nodes_file, output_pml_file):
        """
        Generates a PyMOL script to highlight residues corresponding to nodes from a file.

        Args:
            pdb_file (str): Path to the PDB file.
            atom_mapping (str): Path to the PDB atom mapping file.
            nodes_file (str): Path to the file containing node indices or names.
            output_pml_file (str): Path to the output PyMOL script file.
        """
        # Parse the PDB file to get the structure
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('structure', pdb_file)

        # Read the node names from the file
        try:
            with open(nodes_file, 'r') as f:
                next(f)  # Skip the first line
                node_names = [line.strip() for line in f]
        except FileNotFoundError:
            print(f"Error: The nodes file {nodes_file} was not found.")
            return
        except Exception as e:
            print(f"Error reading nodes file {nodes_file}: {e}")
            return

        # Parse the atom mapping file
        residues = ReadFiles.parse_mapping(atom_mapping)
        residue_dict = {(res_num, chain_id): atom_name for chain_id, res_num, atom_name in residues}

        # Convert node names to residue numbers (adjust as needed)
        try:
            residue_numbers = []
            for node_name in node_names:
                parts = node_name.split()
                if len(parts) > 1:
                    residue_numbers.append(int(parts[1]) + 1)  # Convert the numeric part to integer and adjust
                else:
                    print(f"Warning: Node entry '{node_name}' does not have a valid numeric part.")
                    return
        except ValueError:
            print("Error: One or more entries in the nodes file are not valid integers.")
            return

        # Generate the PyMOL script
        try:
            with open(output_pml_file, 'w') as f:
                f.write(f"load {pdb_file}, structure\n")
                f.write("set cartoon_color, grey90\n")

                # Define a color for the spheres
                f.write("set_color highlight_color, [1.0, 0.0, 0.0]\n")

                # Show residues as spheres
                for res_num in residue_numbers:
                    for model in structure:
                        for chain in model:
                            #print(residue_dict)
                            atom_name = residue_dict.get((res_num, chain.id), 'CA')
                            #print(atom_name)
                            f.write(f"select chain {chain.id} and resi {res_num} and name {atom_name}\n")
                            f.write(f"show spheres, chain {chain.id} and resi {res_num} and name {atom_name}\n")
                            f.write(f"color highlight_color, chain {chain.id} and resi {res_num}\n")

                f.write("set sphere_scale, 0.7\n")  # Adjust sphere size as needed
                f.write("bg_color white\n")

            print(f" 🧊  PyMOL script to highlight top nodes saved to {output_pml_file}")
        except Exception as e:
            print(f"Error writing PyMOL script to {output_pml_file}: {e}")

    def write_pml_script_for_residue_paths(self, residue_list, output_pml_file):
        """
        Generates a PyMOL script to draw lines connecting consecutive residues.

        Args:
            residue_list (list): A list of residue numbers in the order to connect.
            output_pml_file (str): Path to the output PyMOL script file.
        """
        with open(output_pml_file, 'w') as f:
            f.write(f"load {self.pdb_file}, structure\n")  # Ensure you replace `pdb_file` with your actual PDB file path
            f.write("set cartoon_color, grey90\n")
            # Define color and background
            f.write("set_color black, [0.0, 0.0, 0.0]\n")
            f.write("bg_color white\n")

            # Draw lines between consecutive residues
            for i in range(len(residue_list) - 1):
                res1 = residue_list[i]
                res2 = residue_list[i + 1]

                # Create selection for residues
                f.write(f"select resi {res1}, resi {res2}\n")

                # Draw distance and hide label
                f.write(f"distance path_{res1}_{res2}, resi {res1}, resi {res2}\n")
                f.write(f"set gap_width, 0, path_{res1}_{res2}\n")
                f.write(f"color black, path_{res1}_{res2}\n")
                f.write(f"hide labels, path_{res1}_{res2}\n")

        print(f" 🧊  PyMOL script for residue paths saved to {output_pml_file}")

    def write_pml_script_for_top_shortest_paths(self, top_50_file, edge_betweenness_file, output_pml_file):
        """
        Generates a PyMOL script to draw lines connecting consecutive residues in the top shortest paths.
        Edge thickness is based on edge betweenness values.

        Args:
            top_50_file (str): Path to the file containing top shortest paths and mapped residues.
            edge_betweenness_file (str): Path to the file containing edge betweenness values.
            output_pml_file (str): Path to the output PyMOL script file.
        """

        # Load edge betweenness values
        edge_betweenness = ReadFiles.read_edge_betweenness_from_file(edge_betweenness_file)
        #print(edge_betweenness)
        if not edge_betweenness:
            print("Warning: No edge betweenness values found in edge betweenness file.")
        #print("Edges in edge_betweenness:", edge_betweenness.keys())

        max_betweenness = max(edge_betweenness.values(), default=1)  # Avoid division by zero

        # Read paths and mapped paths from the file
        paths = []
        try:
            with open(top_50_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    # Check for Path and Path_mapped
                    if 'Path = [' in line:
                        path_info = re.search(r'Path = \[(.*?)\]', line)
                        if path_info:
                            path_str = path_info.group(1).strip()
                            path = [int(node.strip()) for node in path_str.split('->')]
                            paths.append(path)
        except Exception as e:
            print(f"Error reading top shortest paths file: {e}")
            return


        # Open the output file for writing
        try:
            with open(output_pml_file, 'w') as f:
                # Track unique commands to avoid repetition
                written_selections = set()
                written_distances = set()
                written_spheres = set()
                # Load the PDB file
                f.write(f"load {self.pdb_file}\n")
                f.write(f"set cartoon_color, grey80\n")

                # Process each path and corresponding mapped residues
                for path in paths:
                    #print(path, type(path), type(paths))
                    for i in range(0,len(path)-1):
                        #print(node, type(node))
                        node1 = path[i]
                        node2 = path[i+1]

                        res_name1, atom_name1, res_num1, chain_id1 = self.atom_mapping[str(node1)]
                        res_name2, atom_name2, res_num2, chain_id2 = self.atom_mapping[str(node2)]
                        
                        #betweenness_value = edge_betweenness.get(node1, edge_betweenness.get(node2, None))
                        betweenness_value = edge_betweenness.get((node1, node2)) or edge_betweenness.get((node2, node1))
                        if betweenness_value is not None:
                            norm_betweenness = betweenness_value / max_betweenness
                            thickness = 1 + 10 * norm_betweenness
                            
                            # Generate selection commands only if not already written
                            selection1 = f"select resi_{res_num1}, chain {chain_id1} and resi {res_num1} and name {atom_name1}"
                            selection2 = f"select resi_{res_num2}, chain {chain_id2} and resi {res_num2} and name {atom_name2}"
                    
                            if selection1 not in written_selections:
                                f.write(f"{selection1}\n")
                                written_selections.add(selection1)
                            
                            if selection2 not in written_selections:
                                f.write(f"{selection2}\n")
                                written_selections.add(selection2)
                            
                            # Generate sphere display command only if not already written
                            sphere_cmd = (
                                f"show spheres, resi_{res_num1} and chain {chain_id1} and name {atom_name1} or "
                                f"resi_{res_num2} and chain {chain_id2} and name {atom_name2}"
                            )
                            if sphere_cmd not in written_spheres:
                                f.write(f"{sphere_cmd}\n")
                                written_spheres.add(sphere_cmd)
                    
                            # Generate distance command only if not already written
                            # Sort res_num to ensure consistent ordering
                            sorted_res = sorted([(res_num1, chain_id1, atom_name1), (res_num2, chain_id2, atom_name2)])
                            distance_key = f"edge_{sorted_res[0][0]}_{sorted_res[1][0]}"
                            distance_cmd = (
                                f"distance {distance_key}, "
                                f"chain {sorted_res[0][1]} and resi {sorted_res[0][0]} and name {sorted_res[0][2]}, "
                                f"chain {sorted_res[1][1]} and resi {sorted_res[1][0]} and name {sorted_res[1][2]}"
                            )
                    
                            if distance_cmd not in written_distances:
                                f.write(f"{distance_cmd}\n")
                                f.write(f"set dash_width, {thickness:.2f}, {distance_key}\n")
                                written_distances.add(distance_cmd)
                        else:
                            print(f"Warning: Edge key ({path[i]}, {path[i + 1]}) or ({path[i + 1]}, {path[i]}) not found in edge_betweenness.")
        
                # Finalize the PyMOL script
                # Finalize the PyMOL script
                f.write("hide labels \n")
                f.write("set dash_gap, 0 \n")
                f.write("set dash_color, grey10 \n")
                f.write("bg_color white\n")
                #f.write("select backbone, name P or name CA\n")
                #f.write("cmd.show_as('spheres', 'backbone')\n")
                f.write("set sphere_transparency, 0.3 \n")

            print(f" 🧊  PyMOL script for top shortest paths saved to {output_pml_file}")

        except Exception as e:
            print(f"Error writing PyMOL script to file: {e}")

    def write_pml_script_for_alternative_paths(self, alternative_paths_file, output_pml_file):
        """
        Generates a PyMOL script to draw lines connecting consecutive residues in the top shortest paths.
        Edge thickness is based on edge betweenness values.

        Args:
            top_50_file (str): Path to the file containing top shortest paths and mapped residues.
            edge_betweenness_file (str): Path to the file containing edge betweenness values.
            output_pml_file (str): Path to the output PyMOL script file.
        """


        # Read paths and mapped paths from the file
        paths = []
        try:
            with open(alternative_paths_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    # Check for Path and Path_mapped
                    #print(line)
                    if 'Path = [' in line:
                        path_info = re.search(r'Path = \[(.*?)\]', line)
                        if path_info:
                            path_str = path_info.group(1).strip()
                            path = [int(node.strip()) for node in path_str.split('->')]
                            paths.append(path)
        except Exception as e:
            print(f"Error reading alternative paths paths file: {e}")
            return


        # Open the output file for writing
        try:
            with open(output_pml_file, 'w') as f:
                # Load the PDB file
                f.write(f"load {self.pdb_file}\n")

                # Process each path and corresponding mapped residues
                for path in paths:
                    #print(path, type(path), type(paths))
                    for i in range(0,len(path)-1):
                        #print(node, type(node))
                        node1 = path[i]
                        node2 = path[i+1]

                        res_name1, atom_name1, res_num1, chain_id1 = self.atom_mapping[str(node1)]
                        res_name2, atom_name2, res_num2, chain_id2 = self.atom_mapping[str(node2)]
                        f.write(f"select resi_{res_num1}, chain {chain_id1} and resi {res_num1} and name {atom_name1}\n")
                        f.write(f"select resi_{res_num2}, chain {chain_id2} and resi {res_num2} and name {atom_name2}\n")
                        f.write(f"show spheres, resi_{res_num1} and chain {chain_id1} and name {atom_name1} or resi_{res_num2} and chain {chain_id2} and name {atom_name2}\n")
                        #f.write(f"show spheres, resi_{res_num1} and name {atom_name1} or resi_{res_num2} and name {atom_name2}\n")
                        f.write(f"distance edge_{res_num1}_{res_num2}, chain {chain_id1} and resi {res_num1} and name {atom_name1}, chain {chain_id2} and resi {res_num2} and name {atom_name2}\n")
                        
                # Finalize the PyMOL script
                f.write("hide labels \n")
                f.write("set dash_gap, 0 \n")
                f.write("set dash_color, grey10 \n")
                f.write("bg_color white\n")
                #f.write("select backbone, name P or name CA\n")
                #f.write("cmd.show_as('spheres', 'backbone')\n")
                f.write("set sphere_transparency, 0.3 \n")

            #print(f" 🧊  PyMOL script for top shortest paths saved to {output_pml_file}")

        except Exception as e:
            print(f"Error writing PyMOL script to file: {e}")


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
                    community_idx = int(parts[0].split()[1])
                    node = int(parts[1].strip())

                    if community_idx not in communities:
                        communities[community_idx] = []

                    communities[community_idx].append(node)

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
        #print(communities)

        # Initialize the PDB parser and load the structure
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('structure', self.pdb_file)

        # Assign a unique color to each community
        community_colors = {community: [random.random(), random.random(), random.random()] for community in communities}

        with open(output_pml_file, 'w') as f:
            # Write the command to load the structure
            f.write(f"load {self.pdb_file}, structure\n")

            # Define colors for each community
            for community, color in community_colors.items():
                f.write(f"set_color color{community}, [{', '.join(map(str, color))}]\n")

            # Apply colors to residues based on community assignment
            for community, nodes in communities.items():
                color = f"color{community}"
                for node in nodes:
                    res_num = int(node) + 1  # Convert node index to residue number (1-based)
                    for model in structure:
                        for chain in model:
                            for residue in chain:
                                if residue.id[1] == res_num:
                                    f.write(f"color {color}, chain {chain.id} and resi {res_num}\n")

            # Show the structure as cartoon
            f.write("show cartoon\n")

        print(f"PyMOL script for communities saved to {output_pml_file}")

    def graph_pml(self, atom_mapping, centrality_file, edge_betweenness_file, output_pml):
        """
        Generates PyMOL scripts to visualize the graph with centrality and edge betweenness.

        Args:
            atom_mapping (str): Path to the PDB atom mapping file.
            centrality_file (str): Path to the file containing centrality values.
            edge_betweenness_file (str): Path to the file containing edge betweenness values.
            output_pml (str): Prefix for the output PyMOL script files.
        """
        # Read the atom mapping
        print(atom_mapping)
        residues = ReadFiles.parse_mapping(atom_mapping)
        print(residues)
        #for i in range(0, len(residues)): print(i, residues[i])
        if not residues:
            print("Warning: No residues found in the atom mapping file.")

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
            if node < len(residues):
                chain_id, res_num, atom_name = residues[node]
                norm_centrality = centrality_value / max_centrality
                sphere_scale = 1 * norm_centrality  # Simplified sphere scale calculation

                for file in output_files.values():
                    file.write(f"show spheres, chain {chain_id} and resi {res_num} and name {atom_name}\n")
                    file.write(f"set sphere_scale, {sphere_scale:.2f}, chain {chain_id} and resi {res_num} and name {atom_name}\n")
            else:
                print(f"Warning: Node {node} is out of bounds for the residue list.")

        # Draw edges with thickness based on edge betweenness
        max_betweenness = max(edge_betweenness.values(), default=1)  # Avoid division by zero
        for (node1, node2), betweenness_value in edge_betweenness.items():
            if node1 < len(residues) and node2 < len(residues):
                chain_id1, res_num1, atom_name1 = residues[node1]
                chain_id2, res_num2, atom_name2 = residues[node2]

                #print(chain_id1,"ch_id",res_num1,"res_num")

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
            else:
                print(f"Warning: Nodes {node1} or {node2} are out of bounds for the residue list.")

        for file in output_files.values():
            file.write("hide labels \n")
            file.write("set dash_gap,0 \n")
            file.write("set dash_color, grey10 \n")
            file.write("bg_color white\n")
            file.close()

        print(f"PyMOL script for graph attributes saved with prefix {output_pml}")

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

    def cliques_pml(self, atom_mapping, cliques_file, edge_betweenness_file, output_pml):
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
        edge_betweenness = ReadFiles.read_edge_betweenness_from_file(edge_betweenness_file)
        residues = ReadFiles.parse_mapping(atom_mapping)
        residue_dict = {(res_num, chain_id): atom_name for chain_id, res_num, atom_name in residues}

        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('structure', self.pdb_file)

        # Assign a unique color to each clique and generate PyMOL commands
        with open(output_pml, 'w') as f:
            f.write(f"load {self.pdb_file}, structure\n")
            f.write("show cartoon\n")

            for i, (clique, nodes) in enumerate(cliques.items(), start=1):
                color = [random.random(), random.random(), random.random()]
                f.write(f"set_color clique_{i}, [{', '.join(map(str, color))}]\n")

                # Create a selection for this clique
                selection_residues = []
                for node in nodes:
                    res_num = int(node) + 1  # Convert node index to residue number (1-based)
                    for model in structure:
                        for chain in model:
                            for residue in chain:
                                if residue.id[1] == res_num:
                                    atom_name = residue_dict.get((res_num, chain.id), 'CA')
                                    selection_residues.append(f"chain {chain.id} and resi {res_num}")
                                    f.write(f"color clique_{i}, chain {chain.id} and resi {res_num}\n")
                                    f.write(f"show spheres, chain {chain.id} and resi {res_num} and name {atom_name}\n")

                # Combine all residue selections into one command
                if selection_residues:
                    f.write(f"select clique_{i}, {' + '.join(selection_residues)}\n")

            # Draw edges with thickness based on edge betweenness, but only for nodes in the same clique
            max_betweenness = max(edge_betweenness.values(), default=1)  # Avoid division by zero
            for i, (clique, nodes) in enumerate(cliques.items(), start=1):
                for m in range(len(nodes)):
                    for n in range(m + 1, len(nodes)):
                        node1, node2 = nodes[m], nodes[n]
                        if (node1, node2) in edge_betweenness or (node2, node1) in edge_betweenness:
                            betweenness_value = edge_betweenness.get((node1, node2), edge_betweenness.get((node2, node1), 0))
                            thickness = 1 + 10 * (betweenness_value / max_betweenness)

                            # Safely get atom names, ensuring the keys exist
                            atom_name1 = residue_dict.get((node1 + 1, chain.id), 'CA')  # Default to 'CA' if not found
                            atom_name2 = residue_dict.get((node2 + 1, chain.id), 'CA')

                            for model in structure:
                                for chain in model:
                                    f.write(f"distance edge_{node1}_{node2}, chain {chain.id} and resi {node1+1} and name {atom_name1}, chain {chain.id} and resi {node2+1} and name {atom_name2}\n")
                                    f.write(f"set dash_width, {thickness:.2f}, edge_{node1}_{node2}\n")

            f.write("set dash_gap, 0\n")
            f.write("set dash_color, grey10\n")
            f.write("hide labels\n")
            f.write("bg_color white\n")

        print(f"PyMOL script for cliques saved to {output_pml}")



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

                # Define a color for the spheres
                f.write("set_color highlight_color, [1.0, 0.0, 0.0]\n")

                # Show residues as spheres
                for res_num in residue_numbers:
                    for model in structure:
                        for chain in model:
                            #print(residue_dict)
                            atom_name = residue_dict.get((res_num, chain.id), 'CA')
                            #print(atom_name)
                            f.write(f"select chain {chain.id} and resi_{res_num} and name {atom_name}\n")
                            f.write(f"show spheres, chain {chain.id} and resi {res_num} and name {atom_name}\n")
                            f.write(f"color highlight_color, chain {chain.id} and resi {res_num}\n")

                f.write("set sphere_scale, 0.5\n")  # Adjust sphere size as needed

            print(f"PyMOL script to highlight top nodes saved to {output_pml_file}")
        except Exception as e:
            print(f"Error writing PyMOL script to {output_pml_file}: {e}")

    def write_pml_script_for_residue_paths(residue_list, output_pml_file):
        """
        Generates a PyMOL script to draw lines connecting consecutive residues.

        Args:
            residue_list (list): A list of residue numbers in the order to connect.
            output_pml_file (str): Path to the output PyMOL script file.
        """
        with open(output_pml_file, 'w') as f:
            f.write(f"load {pdb_file}, structure\n")  # Ensure you replace `pdb_file` with your actual PDB file path

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

        print(f"PyMOL script for residue paths saved to {output_pml_file}")

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
        if not edge_betweenness:
            print("Warning: No edge betweenness values found in edge betweenness file.")
        #print("Edges in edge_betweenness:", edge_betweenness.keys())

        max_betweenness = max(edge_betweenness.values(), default=1)  # Avoid division by zero

        # Read paths and mapped paths from the file
        paths = []
        paths_mapped = []
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

                    elif 'Path_mapped =' in line:
                        mapped_info = re.search(r'Path_mapped = \[(.*?)\]', line)
                        if mapped_info:
                            mapped_str = mapped_info.group(1).strip()
                            mapped_residues = [residue.strip() for residue in mapped_str.split('->')]
                            paths_mapped.append(mapped_residues)
        except Exception as e:
            print(f"Error reading top shortest paths file: {e}")
            return

        if not paths or not paths_mapped:
            print("Warning: No paths or mapped paths extracted from the top shortest paths file.")

        # Open the output file for writing
        try:
            with open(output_pml_file, 'w') as f:
                # Load the PDB file
                f.write(f"load {self.pdb_file}\n")

                # Process each path and corresponding mapped residues
                for path, mapped_path in zip(paths, paths_mapped):
                    if len(path) != len(mapped_path):
                        print(f"Warning: Length mismatch between path and mapped path.")
                        continue

                    for i in range(len(path) - 1):
                        residue1 = mapped_path[i]
                        residue2 = mapped_path[i + 1]

                        # Extract residue number and atom name from residue1 and residue2
                        match1 = re.search(r'Residue (\d+)\s(\w+)(?:\s(\w+))?', residue1)
                        match2 = re.search(r'Residue (\d+)\s(\w+)(?:\s(\w+))?', residue2)
                        #print(match1)

                        if match1 and match2:
                            res_num1, atom_name1, chain1 = match1.groups()
                            res_num2, atom_name2, chain2 = match2.groups()

                            chain1 = chain1 or ' '
                            chain2 = chain2 or ' '

                            # Convert edge format to match edge_betweenness
                            edge_key1 = (path[i], path[i + 1])
                            edge_key2 = (path[i + 1], path[i])

                            betweenness_value = edge_betweenness.get(edge_key1, edge_betweenness.get(edge_key2, None))

                            if betweenness_value is not None:
                                norm_betweenness = betweenness_value / max_betweenness
                                thickness = 1 + 10 * norm_betweenness

                                # Write PyMOL script lines
                                f.write(f"select resi_{res_num1}, chain {chain1} and resi {res_num1} and name {atom_name1}\n")
                                f.write(f"select resi_{res_num2}, chain {chain2} and resi {res_num2} and name {atom_name2}\n")
                                f.write(f"show spheres, resi_{res_num1} and chain {chain1} and name {atom_name1} or resi_{res_num2} and chain {chain2} and name {atom_name2}\n")
                                #f.write(f"show spheres, resi_{res_num1} and name {atom_name1} or resi_{res_num2} and name {atom_name2}\n")
                                f.write(f"distance edge_{res_num1}_{res_num2}, chain {chain1} and resi {res_num1} and name {atom_name1}, chain {chain2} and resi {res_num2} and name {atom_name2}\n")
                                f.write(f"set dash_width, {thickness:.2f}, edge_{res_num1}_{res_num2}\n")
                            else:
                                print(f"Warning: Edge key ({path[i]}, {path[i + 1]}) or ({path[i + 1]}, {path[i]}) not found in edge_betweenness.")
                        else:
                            print(f"Error processing residues {residue1} and {residue2}: Unable to extract residue number and atom name.")

                # Finalize the PyMOL script
                f.write("hide labels \n")
                f.write("set dash_gap, 0 \n")
                f.write("set dash_color, grey10 \n")
                f.write("bg_color white\n")
                #f.write("select backbone, name P or name CA\n")
                #f.write("cmd.show_as('spheres', 'backbone')\n")
                f.write("set sphere_transparency, 0.3 \n")

            print(f"PyMOL script for top shortest paths saved to {output_pml_file}")

        except Exception as e:
            print(f"Error writing PyMOL script to file: {e}")

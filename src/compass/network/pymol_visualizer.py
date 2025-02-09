import random
import re

from Bio import PDB

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
                    community_idx = parts[0].split()[1]
                    node_list = parts[1].strip()
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
            f.write(f"load {self.pdb_file} \n")

            # Define colors for each community
            for community, color in community_colors.items():
                f.write(
                    f"set_color color{community}, [{', '.join(map(str, color))}]\n")
            # Apply colors to residues based on community assignment
            for community, nodes in communities.items():
                color = f"color{community}"
                nodes = [num for num in nodes[0].split(',')]
                for node in nodes:
                    chain_id, res_num = node.split('_')[
                                        :2]  # Splitting on '_' and assuming chain_id and res_num are in this format
                    if chain_id:  # If chain_id is not empty
                        f.write(
                            f"color {color}, chain {chain_id} and resi {res_num}\n")
                    else:  # If there's no chain_id (for standalone residues?)
                        f.write(f"color {color}, resi {res_num}\n")
                        # Show the structure as cartoon

            f.write("show cartoon\n")
            f.write("bg_color white\n")

        print(f" 🧊  PyMOL script for communities saved to {output_pml_file}")

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
                if len(parts) != 2:
                    continue  # Skip invalid lines
                clique_idx = int(parts[0].split()[1])  # Extract clique index
                nodes_list = parts[1].split()  # Clean and convert nodes
                cliques[clique_idx] = nodes_list

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
            f.write(f"load {self.pdb_file}\n")
            f.write("show cartoon\n")
            f.write("set cartoon_color, grey90\n")

            for i, (clique, nodes) in enumerate(cliques.items(), start=1):
                color = [random.random(), random.random(), random.random()]
                f.write(
                    f"set_color clique_{i}, [{', '.join(map(str, color))}]\n")
                # Create a selection for this clique
                selection_residues = []
                for node in nodes:
                    chain_id, res_num = node.split('_')[:2]
                    res_num = int(res_num.strip().replace(',', ''))
                    selection_residues.append(f"chain {chain_id} and resi {res_num}")
                    f.write(f"color clique_{i}, chain {chain_id} and resi {res_num}\n")
                    f.write(f"show spheres, chain {chain_id} and resi {res_num} and (name CA or name C5')\n")
                # Combine all residue selections into one command
                if selection_residues:
                    f.write(
                        f"select clique_{i}, {' + '.join(selection_residues)}\n")
            f.write("set dash_gap, 0\n")
            f.write("set dash_color, grey10\n")
            f.write("hide labels\n")
            f.write("set sphere_scale, 0.7\n")
            f.write("bg_color white\n")

        print(f" 🧊  PyMOL script for cliques saved to {output_pml}")

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
        centrality_df = ReadFiles.read_centrality_from_file(centrality_file)
        betweenness_df = ReadFiles.read_edge_betweenness_from_file(
            edge_betweenness_file)
        output_files = {
            "all": open(f"{output_pml}_all.pml", 'w'),
        }

        # Write initial structure loading and visualization commands
        for file in output_files.values():
            file.write(f"load {self.pdb_file}\n")
            file.write(f"show_as cartoon, structure\n")
            file.write(f"set cartoon_transparency, 0.6\n")

        # Set sphere size based on centrality
        max_centrality = centrality_df[
            'Betweenness'].max()  # Avoid division by zero
        for _, row in centrality_df.iterrows():
            res_num = row['Node_Res_Num']
            chain_id = row['Chain_ID']
            centrality_value = row['Betweenness']
            norm_centrality = centrality_value / max_centrality
            sphere_scale = 0.3 + 1 * norm_centrality  # Simplified sphere scale calculation
            for file in output_files.values():
                file.write(
                    f"show spheres, chain {chain_id} and resi {res_num} and (name CA or name C5')\n")
                file.write(
                    f"set sphere_scale, {sphere_scale:.2f}, chain {chain_id} and resi {res_num} and (name CA or name C5')\n")

        # Draw edges with thickness based on edge betweenness
        max_betweenness = betweenness_df[
            'Betweenness'].max()  # Avoid division by zero
        for _, row in betweenness_df.iterrows():
            # print(row.keys())
            res1 = int(row['Res1'])
            res2 = int(row['Res2'])
            chain1 = str(row['Chain1'])
            chain2 = str(row['Chain2'])
            betweenness_value = row['Betweenness']

            norm_betweenness = betweenness_value / max_betweenness
            thickness = 0.5 + 10 * norm_betweenness

            output_files["all"].write(
                f"distance edge_{res1}_{res2}, chain {chain1} and resi {res1} and (name CA or name C5'), chain {chain2} and resi {res2} and (name CA or name C5')\n"
            )
            output_files["all"].write(
                f"set dash_width, {thickness:.2f}, edge_{res1}_{res2}\n"
            )

        # Final settings for PyMOL visualization
        for file in output_files.values():
            file.write("hide labels\n")
            file.write("set dash_gap, 0\n")
            file.write("set dash_color, black\n")
            file.write("bg_color white\n")
            file.close()

        print(
            f" 🧊  PyMOL script for graph attributes saved with prefix {output_pml}")

    def highlight_top_nodes_pml(self, pdb_file, atom_mapping, nodes_file,
                                output_pml_file):
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

        try:
            # Read node information from the file
            residue_info = []
            with open(nodes_file, 'r') as f:
                for line in f:
                    if line.strip().startswith("Node"):
                        # Extract residue number and chain ID
                        node_data = line.split("Node (")[1].split(")")[
                            0].split(",")
                        res_num = int(node_data[0].strip())
                        chain_id = node_data[1].strip()
                        residue_info.append((res_num, chain_id))

            # Generate the PyMOL script
            with open(output_pml_file, 'w') as f:
                f.write(f"load {pdb_file}\n")
                f.write("set cartoon_color, grey90\n")

                # Define a color for the spheres
                f.write("set_color highlight_color, [1.0, 0.0, 0.0]\n")
                residue_selections = []
                #selection_string = "sele hotspot_residues, "

                # Highlight residues as spheres
                for res_num, chain_id in residue_info:
                    selection = f"(chain {chain_id} and resi {res_num} and (name CA or name C5'))"
                    residue_selections.append(selection)
                    # Show spheres and color each residue
                    f.write(f"show spheres, {selection}\n")
                    f.write(f"color highlight_color, {selection}\n")

                selection_string = f"sele hotspot_residues, {' or '.join(residue_selections)}\n"
                f.write(selection_string)
                # Set sphere scale and background color
                f.write("set sphere_scale, 0.7\n")
                f.write("bg_color white\n")

            print(
                f" 🧊 PyMOL script to highlight top nodes saved to {output_pml_file}")

        except FileNotFoundError as e:
            print(f"Error: File not found - {e}")

    def write_pml_script_for_residue_paths(self, residue_list,
                                           output_pml_file):
        """
        Generates a PyMOL script to draw lines connecting consecutive residues.

        Args:
            residue_list (list): A list of residue numbers in the order to connect.
            output_pml_file (str): Path to the output PyMOL script file.
        """
        with open(output_pml_file, 'w') as f:
            f.write(
                f"load {self.pdb_file}\n")  # Ensure you replace `pdb_file` with your actual PDB file path
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
                f.write(
                    f"distance path_{res1}_{res2}, resi {res1}, resi {res2}\n")
                f.write(f"set gap_width, 0, path_{res1}_{res2}\n")
                f.write(f"color black, path_{res1}_{res2}\n")
                f.write(f"hide labels, path_{res1}_{res2}\n")

        print(f" 🧊  PyMOL script for residue paths saved to {output_pml_file}")

    # def read_edge_betweenness_file(edge_betweenness_file):

    def write_pml_script_for_top_shortest_paths(self, top_50_file,
                                                edge_betweenness_file,
                                                output_pml_file):
        """
        Generates a PyMOL script to draw lines connecting consecutive residues in the top shortest paths.
        Edge thickness is based on edge betweenness values.

        Args:
            top_50_file (str): Path to the file containing top shortest paths and mapped residues.
            edge_betweenness_file (str): Path to the file containing edge betweenness values.
            output_pml_file (str): Path to the output PyMOL script file.
        """
        try:
            # Read edge betweenness data
            betweenness_df = ReadFiles.read_edge_betweenness_from_file(
                edge_betweenness_file)
            max_betweenness = betweenness_df['Betweenness'].max()
        except Exception as e:
            print(f"Error reading edge betweenness file: {e}")
            return

        # Parse paths from the input file
        paths = []
        try:
            with open(top_50_file, 'r') as f:
                for line in f:
                    if 'Path = [' in line:
                        match = re.search(r'Path = \[(.*?)\]', line)
                        if match:
                            path_str = match.group(1).strip()
                            path = [int(node.strip()) for node in
                                    path_str.split('->')]
                            paths.append(path)
        except Exception as e:
            print(f"Error reading top shortest paths file: {e}")
            return

        try:
            # Open the output PyMOL script file
            with open(output_pml_file, 'w') as f:
                written_selections = set()
                written_distances = set()
                written_spheres = set()
                # Load the PDB file
                f.write(f"load {self.pdb_file}\n")
                # Process paths
                for path in paths:
                    for i in range(len(path) - 1):
                        node1 = path[i]
                        node2 = path[i + 1]
                        # Atom mappings
                        try:
                            res_name1, atom_name1, res_num1, chain_id1 = \
                            self.atom_mapping[str(node1)]
                            res_name2, atom_name2, res_num2, chain_id2 = \
                            self.atom_mapping[str(node2)]
                        except KeyError as e:
                            print(
                                f"Warning: Node {e} not found in atom mappings.")
                            continue

                        # Edge betweenness
                        edge_betweenness_row = betweenness_df[
                            ((betweenness_df['Res1'] == res_num1) & (
                                        betweenness_df['Res2'] == res_num2)) |
                            ((betweenness_df['Res1'] == res_num2) & (
                                        betweenness_df['Res2'] == res_num1))
                            ]
                        if edge_betweenness_row.empty:
                            print(
                                f"Warning: No betweenness data for edge ({node1}, {node2}).")
                            continue

                        betweenness_value = edge_betweenness_row.iloc[0][
                            'Betweenness']
                        norm_betweenness = betweenness_value / max_betweenness
                        thickness = 1 + 10 * norm_betweenness

                        # Selection commands
                        selection1 = f"select resi_{res_num1}, chain {chain_id1} and resi {res_num1} and name {atom_name1}"
                        selection2 = f"select resi_{res_num2}, chain {chain_id2} and resi {res_num2} and name {atom_name2}"
                        if selection1 not in written_selections:
                            f.write(f"{selection1}\n")
                            written_selections.add(selection1)
                        if selection2 not in written_selections:
                            f.write(f"{selection2}\n")
                            written_selections.add(selection2)

                        # Sphere commands
                        sphere_cmd1 = f"show spheres, resi {res_num1} and chain {chain_id1} and name {atom_name1}"
                        sphere_cmd2 = f"show spheres, resi {res_num2} and chain {chain_id2} and name {atom_name2}"
                        if sphere_cmd1 not in written_spheres:
                            f.write(f"{sphere_cmd1}\n")
                            written_spheres.add(sphere_cmd1)
                        if sphere_cmd2 not in written_spheres:
                            f.write(f"{sphere_cmd2}\n")
                            written_spheres.add(sphere_cmd2)

                        # Distance commands
                        sorted_res = sorted([(res_num1, chain_id1, atom_name1),
                                             (
                                             res_num2, chain_id2, atom_name2)])
                        distance_key = f"edge_{sorted_res[0][0]}_{sorted_res[1][0]}"
                        distance_cmd = (
                            f"distance {distance_key}, "
                            f"chain {sorted_res[0][1]} and resi {sorted_res[0][0]} and name {sorted_res[0][2]}, "
                            f"chain {sorted_res[1][1]} and resi {sorted_res[1][0]} and name {sorted_res[1][2]}"
                        )
                        if distance_cmd not in written_distances:
                            f.write(f"{distance_cmd}\n")
                            f.write(
                                f"set dash_width, {thickness:.2f}, {distance_key}\n")
                            written_distances.add(distance_cmd)

                # Finalize script
                f.write("hide labels\n")
                f.write("set dash_gap, 0\n")
                f.write("set dash_color, grey10\n")
                f.write("bg_color white\n")
                f.write("set sphere_transparency, 0.3\n")
                f.write("set sphere_scale, 0.5\n")

            print(f"🧊 PyMOL script for top shortest paths saved to {output_pml_file}")
        except Exception as e:
            print(f"Error writing PyMOL script to file: {e}")

    def write_pml_script_for_alternative_paths(self, alternative_paths_file,
                                               output_pml_file):
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
                    if 'Path' in line or 'Alternative Path' in line:
                        # Extract the path from the line
                        path_info = re.search(r':\s*(.*)', line)
                        if path_info:
                            path_str = path_info.group(1).strip()
                            # Parse residue-chain pairs (e.g., "1060,A")
                            path = [
                                tuple(node.strip().split(','))
                                for node in path_str.split('->')
                                if ',' in node
                            ]
                            paths.append(path)
        except Exception as e:
            print(f"Error reading alternative paths file: {e}")
            return

        # Open the output file for writing
        try:
            with open(output_pml_file, 'w') as f:
                # Load the PDB file
                f.write(f"load {self.pdb_file}\n")
                # Process each path and corresponding residues
                for path in paths:
                    for i in range(len(path) - 1):
                        res_num1, chain_id1 = path[i]
                        res_num2, chain_id2 = path[i + 1]
                        # Write PyMOL commands for selections and visualization
                        f.write(
                            f"select resi_{res_num1}, chain {chain_id1} and resi {res_num1} and (name CA or name C5')\n")
                        f.write(
                            f"select resi_{res_num2}, chain {chain_id2} and resi {res_num2} and (name CA or name C5')\n")
                        f.write(
                            f"show spheres, chain {chain_id1} and resi {res_num1} and (name CA or name C5') or chain {chain_id2} and resi {res_num2} and (name CA or name C5')\n")
                        f.write(
                            f"distance edge_{res_num1}_{res_num2}, "
                            f"chain {chain_id1} and resi {res_num1} and (name CA or name C5'), "
                            f"chain {chain_id2} and resi {res_num2} and (name CA or name C5')\n"
                        )
                # Finalize the PyMOL script
                f.write("hide labels \n")
                f.write("set dash_gap, 0 \n")
                f.write("set dash_color, grey10 \n")
                f.write("bg_color white\n")
                f.write("set sphere_transparency, 0.3 \n")
        except Exception as e:
            print(f"Error writing PyMOL script to file: {e}")

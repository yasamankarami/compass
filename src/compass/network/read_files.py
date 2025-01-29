import json
import numpy as np
from Bio import PDB
import networkx as nx
import pandas as pd 
import mdtraj as md


class ReadFiles:
    """
    A class to handle reading and parsing of files for network analysis, including matrices, structures, and centrality values.
    """

    def read_matrix(self, file_path):
        """
        Reads a matrix from a .txt file.

        Args:
            file_path (str): Path to the matrix file.

        Returns:
            np.ndarray: The matrix read from the file.
        """
        return np.loadtxt(file_path)
        
    def atom_mapping(self, file_path):
        """
        Extracts CA atoms for amino acids and P or O5' atoms for nucleic acids from a topology.

        Returns:
            tuple: A tuple containing:
                - atom_mapping (dict): Mapping of atom indices to atom information.
                - atoms (list): List of atom tuples (residue name, atom name, residue id, chain id).
        """
        # Load the PDB file using MDTraj
        trajectory = md.load(file_path)
        topology = trajectory.topology
        
        ca_atoms = trajectory.topology.select('name CA')
        dna = "(resname =~ '(5|3)?D([ATGC]){1}(3|5)?$')"
        rna = "(resname =~ '(3|5)?R?([AUGC]){1}(3|5)?$')"
        p_atoms = trajectory.topology.select(f'({dna} or {rna}) and name "C5\'"')
        all_atoms = sorted(np.concatenate((ca_atoms, p_atoms)))
    
        atom_mapping = {}  # Maps node index to atom information
        atoms = []
        index_counter = 0
    
        amino_acid_count = 0
        nucleic_acid_count = 0
    
        # Process selected atoms to build atom_mapping
        for atom_index in all_atoms:
            atom = topology.atom(atom_index)
            residue = atom.residue
            chain_id = residue.chain.chain_id if residue.chain.chain_id is not None else ''
            residue_name = residue.name
            residue_id = residue.resSeq
            atom_name = atom.name
        
            if atom_name == 'CA':
                amino_acid_count += 1
            elif atom_name == "C5'":
                nucleic_acid_count += 1
            
            atoms.append((residue_name, atom_name, residue_id, chain_id))
            atom_mapping[index_counter] = (residue_name, atom_name, residue_id, chain_id)
            index_counter += 1

        print(f" 🔍  Processing matrices for graph construction")
        print(f" 📦  Processed {amino_acid_count} amino acid residues.")
        print(f" 🧬  Processed {nucleic_acid_count} nucleic acid residues.")
        print(f" ⚙️   Total residues processed: {len(atoms)}.")
        print(f" 🕸️  Graph network construction is complete.")
        return atom_mapping, atoms

    def parse_mapping(atom_mapping):
        """
        Parses atom mapping to extract residues information.

        Args:
            atom_mapping (dict): A dictionary mapping atom indices to atom information.

        Returns:
            list: A list of tuples (chain_id, res_num, atom_name).
        """
        residues = []
        for index, (res_name, atom_name, res_num, chain_id) in atom_mapping.items():
            residues.append((chain_id, res_num, atom_name))
        return residues
    
    def load_graph_and_mapping(self, input_file):
        """
        Loads the graph and atom mapping from a JSON file.

        Args:
            input_file (str): Path to the JSON file containing the graph and atom mapping.

        Returns:
            tuple: A tuple containing:
                - G (nx.Graph): The loaded graph.
                - atom_mapping (dict): The loaded atom mapping.
        """
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        G = nx.readwrite.json_graph.node_link_graph(data['graph'])
        atom_mapping = data['atom_mapping']
        return G, atom_mapping
        
    def read_centrality_from_file(file_path):
        """
        Processes a file containing node metrics.

        Args:
            file_path (str): Path to the input file.

        Returns:
            pd.DataFrame: A DataFrame containing parsed node metrics.
        """
        data = []
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
            
                # Skip empty lines or headers
                if not line or line.startswith("Node Res_num Chain_ID"):
                    continue
                
                try:
                    # Split the line into components
                    parts = line.split("\t")
                    # Parse the Node column (e.g., "Node (1,A)")
                    node_info = parts[0].split(" ")
                    node_details = node_info[1].strip("()").split(",")
                    #print(node_details)
                    node_res_num = int(node_details[0])  # Extract residue number
                    chain_id = node_details[1]          # Extract chain ID
                    # Parse the remaining columns
                    betweenness = float(parts[1])
                    closeness = float(parts[2])
                    degree = int(parts[3])
                    # Append to the data list
                    data.append({
                        "Node_Res_Num": node_res_num,
                        "Chain_ID": chain_id,
                        "Betweenness": betweenness,
                        "Closeness": closeness,
                        "Degree": degree
                    })
                except (ValueError, IndexError) as e:
                    print(f"Skipping line due to parsing error: {line} ({e})")
    
       
        # Convert data to a Pandas DataFrame for further analysis
        df = pd.DataFrame(data)
        return df

    def read_edge_betweenness_from_file(file_path):
        edges = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines or headers
                if not line or line.startswith("Edge"):
                    continue
                try:
                    # Split the line into edge and betweenness
                    parts = line.split("\t")
                    # Parse the edge column (e.g., "(1,A)-(2,A)")
                    edge_info = parts[0].strip("()").split(")-(")
                    node1 = edge_info[0].strip("()")
                    node2 = edge_info[1].strip("()")
                    # Extract residue numbers and chain IDs
                    res1, chain1 = node1.split(",")
                    res2, chain2 = node2.split(",")
                    # Parse betweenness
                    betweenness = float(parts[1])
                    # Append the data to the list
                    edges.append({
                        "Res1": int(res1),
                        "Chain1": chain1,
                        "Res2": int(res2),
                        "Chain2": chain2,
                        "Betweenness": betweenness
                    })
                except (ValueError, IndexError) as e:
                    print(f"Skipping line due to parsing error: {line} ({e})")
    
        # Convert list of edges to DataFrame
        df = pd.DataFrame(edges)
        return df



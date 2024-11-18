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
        #print(len(ca_atoms), type(ca_atoms))        
        dna = "(resname =~ '(5|3)?D([ATGC]){1}(3|5)?$')"
        rna = "(resname =~ '(3|5)?R?([AUGC]){1}(3|5)?$')"
        p_atoms = trajectory.topology.select(f'({dna} or {rna}) and name "C5\'"')
        #all_atoms = np.concatenate((ca_atoms, c5_atoms))
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
        #print(atom_mapping)
        return atom_mapping, atoms
    '''
    def atom_mapping(self, file_path):
        """
        Reads a structure file (PDB) and extracts CA atoms for amino acids and P or O5' atoms for nucleic acids.

        Args:
            file_path (str): Path to the PDB file.

        Returns:
            tuple: A tuple containing:
                - atom_mapping (dict): Mapping of atom indices to atom information.
                - atoms (list): List of atom objects.
        """
        # Initialize output lists and counter
        atom_mapping = {}  # Maps node index to atom information
        atoms = []
        index_counter = 0
        
        amino_acid_count = 0
        nucleic_acid_count = 0
        missing_residues = set()  # To track missing residues if any

        # Define standard amino acids and nucleotides
        amino_acids = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                       'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                       'THR', 'TRP', 'TYR', 'VAL'}
        nucleotides = {'A', 'T', 'C', 'G', 'U', 'DA', 'DT', 'DC', 'DG', 'DA5', 'DA3', 'DT5', 'DT3', 'DC5', 'DC3', 'DG5', 'DG3'}

        residue_atoms = {}  # Temporary storage for atoms in the current residue

        with open(file_path) as file:
            for line in file.readlines():
                if line.startswith('ATOM'):
                    residue_name = line[17:21].strip()
                    atom_name = line[12:16].strip()
                    chain_id = line[21].strip()
                    residue_id = int(line[22:26].strip())
                    
                    # Create a unique key for each residue
                    residue_key = (residue_name, residue_id, chain_id)
                    
                    # Collect atoms for the current residue
                    if residue_key not in residue_atoms:
                        residue_atoms[residue_key] = set()
                    residue_atoms[residue_key].add(atom_name)

                    # Processing based on residue type
                    if residue_name in amino_acids:
                        if atom_name == 'CA':
                            atoms.append((residue_name, atom_name, residue_id, chain_id))
                            atom_mapping[index_counter] = (residue_name, 'CA', residue_id, chain_id)
                            index_counter += 1
                            amino_acid_count += 1
                    elif residue_name in nucleotides:
                        if atom_name == 'P':
                            atoms.append((residue_name, atom_name, residue_id, chain_id))
                            atom_mapping[index_counter] = (residue_name, 'P', residue_id, chain_id)
                            index_counter += 1
                            nucleic_acid_count += 1
                        elif atom_name == "O5'":
                            if not any(atom_mapping.get(idx)[1] == 'P' for idx in atom_mapping if atom_mapping[idx][2] == residue_id and atom_mapping[idx][3] == chain_id):
                                atoms.append((residue_name, atom_name, residue_id, chain_id))
                                atom_mapping[index_counter] = (residue_name, "O5'", residue_id, chain_id)
                                index_counter += 1
                                nucleic_acid_count += 1

        # Now process non-standard residues that might have C, CA, and N atoms
        for residue_key, atom_set in residue_atoms.items():
            residue_name, residue_id, chain_id = residue_key
            if residue_name not in amino_acids and residue_name not in nucleotides:
                if {'C', 'CA', 'N'}.issubset(atom_set):
                    print(f"Non-standard residue {residue_name} detected with C, CA, and N atoms. Considering it as an amino acid.")
                    atoms.append((residue_name, 'CA', residue_id, chain_id))
                    atom_mapping[index_counter] = (residue_name, 'CA', residue_id, chain_id)
                    index_counter += 1
                else:
                    missing_residues.add(residue_key)

        print(f"Number of amino acid residues processed: {amino_acid_count}")
        print(f"Number of nucleic acid residues processed: {nucleic_acid_count}")
        print(f"Number of atoms extracted: {len(atoms)}")
        print(f"Residues missing 'P' and 'O5\': {sorted(missing_residues)}")
        #print(atom_mapping)
        return atom_mapping, atoms
    '''

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
        Reads centrality values from a file.

        Args:
            file_path (str): Path to the file containing centrality values.

        Returns:
            dict: A dictionary mapping node indices to centrality values.
        """
        centrality = {}
        with open(file_path, 'r') as f:
            # Skip the header line
            next(f)
            for line in f:
                parts = line.strip().split()
                #print(parts)
                if len(parts) == 4:
                    node, betweenness, closeness, degree = parts
                    centrality[int(node)] = float(closeness)  # Save the third value (closeness) as centrality
        return centrality

    def read_edge_betweenness_from_file(file_path):
        """
        Reads edge betweenness values from a file.
    
        Args:
            file_path (str): Path to the file containing edge betweenness values.

        Returns:
            dict: A dictionary mapping edge tuples to betweenness values.
        """
        edge_betweenness = {}
        with open(file_path, 'r') as f:
            # Skip the header line
            next(f)
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:  # Edge and betweenness value
                    edge_str, value = parts
                    node1, node2 = map(int, edge_str.split('-'))
                    edge_betweenness[(node1, node2)] = float(value)
        return edge_betweenness


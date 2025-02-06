# Created by gonzalezroy at 6/6/24
"""Manage common operations as well as those related to topo and traj files"""
import time
from collections import defaultdict
from os.path import basename, join, split

import mdtraj as md
import numpy as np
import prody as prd
from numba import njit
from numba.typed.typeddict import Dict

'''
def remap_toppology(topo, mini_traj, out_dir):
    """
    Remap the topology of a system to start from residue 1 and chain 'A'

    Outputs:
        topo_pdb_name: the renumbered topology in pdb format
        remap_name: the mapping between the original and renumbered residues
        renum_pdb: the renumbered pdb file
    """

    # Save original topology as pdb
    topo_basename = split(topo)[-1].split('.')[0]
    topo_pdb_name = join(out_dir, topo_basename + '_orig.pdb')
    mini_traj.save_pdb(topo_pdb_name)

    # Original info
    parsed = prd.parsePDB(topo_pdb_name)
    nums = parsed.getResnums()
    chains = parsed.getChids()
    segs = [f'"{x}"' if x in {"", ''} else x for x in parsed.getSegnames()]
    orig = sorted(set(zip(nums, chains, segs)), key=lambda x: int(x[0]))

    # After-mapping info
    new_nums = list(range(1, len(orig) + 1))
    parsed.setChids('A')
    new_chains = parsed.getChids()
    new_segs = [f'"{x}"' if x in {"", ''} else x for x in parsed.getSegnames()]
    renum = sorted(set(zip(new_nums, new_chains, new_segs)),
                   key=lambda x: int(x[0]))

    # Output the re-mapping
    remap_name = join(out_dir, topo_basename + '_mapping.txt')
    with open(remap_name, 'w') as f:
        for o, r in zip(orig, renum):
            line = f'{o[0]:>6} {o[1]:>2} {o[2]:>4}  {r[0]:>6} {r[1]:>2} {r[2]:>4}\n'
            f.write(line)

    # Output renumbered pdb
    renum_pdb = topo_pdb_name.replace('_orig.pdb', '_renum.pdb')
    prd.writePDB(renum_pdb, parsed)
'''

def prepare_datastructures(arg, first_timer):
    """
    Prepare datastructures for the calculation of descriptors

    Args:
        arg: namespace with the arguments
        first_timer: first timer to measure the time

    Returns:
        mini_traj: the first chunk of the trajectory
        trajs: list of trajectories
        resids_to_atoms: numba Dict of each residue's indices
        resids_to_noh: numba Dict of each residue's indices without hydrogens
        calphas: numba Dict of each residue's C-alpha indices
        oxy: numba Dict of each residue's oxygen indices
        nitro: numba Dict of each residue's nitrogen indices
        donors: numba Dict of each residue's donor indices
        hydros: numba Dict of each residue's hydrogen indices
        acceptors: numba Dict of each residue's acceptor indices
        corr_indices: list of indices of atoms to be considered for correlation
    """
    # Produce mapping files
    # mapping = Mapping(arg.out_dir)
    # map_file, renumbered_pdb = mapping.pre_processing(arg.topo)
    # mapping.post_processing(map_file, renumbered_pdb)

    # Load trajectory
    trajs = arg.traj.split()
    mini_traj = next(md.iterload(trajs[0], top=arg.topo, chunk=1))
    full_topo = mini_traj.topology.to_dataframe()[0]
    map_file = join(arg.out_dir, 'mapping_file.txt')
    # Produce mapping files
    #remap_toppology(arg.topo, mini_traj, arg.out_dir)

    # Indices of residues in the load trajectory and equivalence
    resids_to_atoms, resids_to_noh, internal_equiv = \
        get_resids_indices(mini_traj)
    #print(resids_to_atoms, "resids_to_atoms in topo_traj")
    raw = {y: x for x in resids_to_atoms for y in resids_to_atoms[x]}
    atoms_to_resids = pydict_to_numbadict(raw)
    
    # Atom selections indices for descriptors calculation
    calphas = get_calpha_p_indices(mini_traj, atoms_to_resids, map_file=map_file)
    oxy, nitro = get_sb_indices(full_topo, atoms_to_resids)
    donors, hydros, acceptors = \
        get_dha_indices(mini_traj, arg.heavies, atoms_to_resids)
    corr_indices_raw = get_calpha_p_indices(mini_traj, atoms_to_resids,map_file=map_file,
                                            numba=False).keys()
    corr_indices = list(corr_indices_raw)
    
    prep_time = round(time.time() - first_timer, 2)
    print(f" 📋 System details: number of trajectories are {len(trajs)}")
    print(f" 📋 System details: number of residues are {len(calphas)}")
    print(f" ⏱️  Until datastructures prepared: {prep_time} s")

    return (
        mini_traj, trajs, resids_to_atoms, resids_to_noh, calphas, oxy, nitro,
        donors, hydros, acceptors, corr_indices)


def get_xyz_chunks(trajs, topo, chunk_size=500):
    """
    Load chunks of xyz coordinates from a list of trajectories

    Args:
        trajs: list of trajectories
        topo: system topology
        chunk_size: size of the chunk to load

    Returns:
        chunk.xyz: chunk of xyz coordinates
    """
    for traj in trajs:
        chunks = md.iterload(traj, top=topo, chunk=chunk_size)
        for chunk in chunks:
            yield chunk.xyz

def get_resids_indices(trajectory):
    """
    Get indices of residues in the loaded trajectory, properly handling protein and nucleic acid residues
    while excluding ions and other non-residue atoms.
    
    Args:
        trajectory: trajectory loaded in mdtraj format
    Returns:
        res_ind_numba: numba Dict of each residue's all atoms indices
        res_ind_noh_numba: numba Dict of each residue's non-hydrogen atom indices
        babel_dict: the equivalence between the original resid numbering and
                   the 0-based numbering used internally
    """
    # Parse the topological information
    df = trajectory.topology.to_dataframe()[0]
    
    # Select CA as backbone atoms of the protein
    ca_atoms = trajectory.topology.select("name CA")
    
    # Select C5' as backbone atoms of the nucleic acids
    dna = "(resname =~ '(5|3)?D([ATGC]){1}(3|5)?$')"
    rna = "(resname =~ '(3|5)?R?([AUGC]){1}(3|5)?$')"
    p_atoms = trajectory.topology.select(f'({dna} or {rna}) and name "C5\'"')
    all_atoms = sorted(np.concatenate((ca_atoms, p_atoms)))
    res_names = [trajectory.topology.atom(i).residue.name for i in all_atoms]
    unique_res_names = np.unique(res_names)
    #print(unique_res_names)
    
    # Create a mask for valid residues (proteins and nucleic acids)
    valid_residues_mask = df['resName'].isin(unique_res_names)
    
    # Filter the dataframe to include only valid residues
    df_filtered = df[valid_residues_mask]
    #print(df_filtered[:-10],"printing df in topo_traj")
    
    # Group by chain, residue number, and segment for valid residues only
    group_by_index = df_filtered.groupby(["chainID", "resSeq", "segmentID"]).indices
    
    # Create non-hydrogen version
    group_by_index_noh = {}
    for key in group_by_index:
        values = group_by_index[key]
        noh = values[df_filtered.loc[values, "element"] != "H"]
        group_by_index_noh[key] = noh
    
    # Create babel dictionaries
    babel_dict = {i: x for i, x in enumerate(group_by_index)}
    
    # Transform to zero-based indices dictionaries
    res_ind_zero = {i: group_by_index[x] for i, x in enumerate(group_by_index)}
    res_ind_noh = {i: group_by_index_noh[x] for i, x in enumerate(group_by_index_noh)}
    
    # Convert to numba dictionaries
    res_ind_numba = pydict_to_numbadict(res_ind_zero)
    res_ind_noh_numba = pydict_to_numbadict(res_ind_noh)
    
    return res_ind_numba, res_ind_noh_numba, babel_dict
'''
def get_resids_indices(trajectory):
    """
    Get indices of residues in the load trajectory

    Args:
        trajectory: trajectory loaded in mdtraj format

    Returns:
        res_ind_numba: numba Dict of each residue's indices
        babel_dict: the equivalence between the original resid numbering and
                    the 0-based used internally
    """
    # Parse the topological information
    df = trajectory.topology.to_dataframe()[0]
    group_by_index = df.groupby(["chainID", "resSeq", "segmentID"]).indices
    group_by_index_noh = {}
    for key in group_by_index:
        values = group_by_index[key]
        noh = values[df.loc[values, "element"] != "H"]
        group_by_index_noh[key] = noh

    babel_dict = {i: x for i, x in enumerate(group_by_index)}
    babel_dict_noh = {i: x for i, x in enumerate(group_by_index_noh)}

    # Transform to numba-dict
    res_ind_zero = {i: group_by_index[x] for i, x in enumerate(group_by_index)}
    res_ind_noh = {i: group_by_index_noh[x] for i, x in
                   enumerate(group_by_index_noh)}

    res_ind_numba = pydict_to_numbadict(res_ind_zero)
    res_ind_noh_numba = pydict_to_numbadict(res_ind_noh)
    return res_ind_numba, res_ind_noh_numba, babel_dict

'''
def get_corr_indices(trajectory, map_file):
    """
    Get atomic indices for correlation calculation

    Args:
        trajectory: trajectory loaded in mdtraj format

    Returns:
        all_atoms: indices of all atoms to be considered for correlation
    """

    # Select CA as backbone atoms of the protein
    ca_atoms = trajectory.topology.select("name CA")

    # Select C5' as backbone atoms of the nucleic acids
    dna = "(resname =~ '(5|3)?D([ATGC]){1}(3|5)?$')"
    rna = "(resname =~ '(3|5)?R?([AUGC]){1}(3|5)?$')"
    p_atoms = trajectory.topology.select(f'({dna} or {rna}) and name "C5\'"')
    #all_atoms = (np.concatenate((ca_atoms, p_atoms)))
    all_atoms = sorted(np.concatenate((ca_atoms, p_atoms)))

    # Write atom details to the specified map_file
    with open(map_file, 'w') as file:
        for idx in all_atoms:
            atom = trajectory.topology.atom(idx)

            # Writing atom details to file
            file.write(f"Atom Index: {idx}, Atom Name: {atom.name}, "
                       f"Residue Name: {atom.residue.name}, Residue Index: {atom.residue.index}, "
                       f"Residue Number: {atom.residue}, chain id:{atom.residue.chain.chain_id}\n")
    return np.asarray(all_atoms, dtype=np.int32)


def get_calpha_p_indices(trajectory, atoms_to_resids, map_file, numba=True ):
    """
    Get atomic indices for C-alpha atoms

    Args:
        trajectory: trajectory loaded in mdtraj format
        atoms_to_resids: dict mapping atoms indices to residues indices
        numba: whether to return a numba dict or a regular dict

    Returns:
        alphas: indices of C-alpha atoms
    """
    # Select CA as backbone atoms of the protein
    ca_atoms = trajectory.topology.select("name CA")

    # Select C5' as backbone atoms of the nucleic acids
    dna = "(resname =~ '(5|3)?D([ATGC]){1}(3|5)?$')"
    rna = "(resname =~ '(3|5)?R?([AUGC]){1}(3|5)?$')"
    p_atoms = trajectory.topology.select(f'({dna} or {rna}) and name "C5\'"')
    all_atoms = (np.concatenate((ca_atoms, p_atoms)))
    n_resids = len(all_atoms)
    calphas_p_raw = get_corr_indices(trajectory, map_file)
    #print(np.shape(calphas_p_raw),n_resids)
    calphas_p = {i: atoms_to_resids[calphas_p_raw[i]] for i in range(n_resids)}

    if len(calphas_p) != n_resids:
        raise ValueError("\nThe number of calphas + P atoms is different from"
                         " the number of residues")

    if numba:
        alphas = pydict_to_numbadict(calphas_p)
    else:
        alphas = calphas_p
    return alphas


def get_sb_indices(topo_df, atoms_to_resids):
    """
    Get atomic indices for salt bridges calculation

    Args:
        topo_df: topology dataframe as returned by MDTraj
        atoms_to_resids: dict mapping atoms indices to residues indices

    Returns:
        o_indices: indices of selected oxygen atoms (see VMD definitions)
        n_indices: indices of selected nitrogen atoms (see VMD definitions)

    """
    # Macro definitions
    sel_O1 = topo_df.resName.isin(["ASP", "GLU"])
    sel_O2 = topo_df.element == "O"
    sel_N1 = topo_df.resName.isin(["ARG", "HIS", "LYS", "HSP"])
    sel_N2 = topo_df.element == "N"

    # Get indices of selected atoms
    o_indices = np.array(topo_df[sel_O1 & sel_O2].index)
    n_indices = np.array(topo_df[sel_N1 & sel_N2].index)

    # Process the oxygen indices to a numba dict
    oxy_raw1 = defaultdict(list)
    [oxy_raw1[atoms_to_resids[x]].append(x) for x in o_indices]
    oxy_raw3 = {x: np.asarray(oxy_raw1[x], dtype=np.int32) for x in oxy_raw1}
    oxy = pydict_to_numbadict(oxy_raw3)

    # Process the nitrogen indices to a numba dict
    nitro_raw1 = defaultdict(list)
    [nitro_raw1[atoms_to_resids[x]].append(x) for x in n_indices]
    nitro_raw3 = {x: np.asarray(nitro_raw1[x], dtype=np.int32) for x in
                  nitro_raw1}
    nitro = pydict_to_numbadict(nitro_raw3)
    return oxy, nitro


def get_dha_indices(trajectory, heavies_elements, atoms_to_resids):
    """
    Get 0-based indices of donors, hydrogens, and acceptors in an MDTraj traj

    Args:
        trajectory: MDTraj trajectory object
        heavies_elements: name of elements considered as heavies
        atoms_to_resids: dict mapping atoms indices to residues indices

    Returns:
        donors: indices of donor atoms (N or O bonded to H)
        hydros: indices of hydrogen atoms (H bonded to N or O)
        heavies: indices of heavy atoms (N or O)
    """
    # Get heavies and hydrogen indices
    df, bonds = trajectory.topology.to_dataframe()
    all_hydrogens = set(df[df.element == "H"].index)

    a_raw1 = set(np.where(df.element.isin(heavies_elements))[0])

    # Find D-H indices
    h_raw1 = []
    d_raw1 = []
    for values in bonds:
        at1 = int(values[0])
        at2 = int(values[1])
        if (at1 in all_hydrogens) and (at2 in a_raw1):
            h_raw1.append(at1)
            d_raw1.append(at2)
        elif (at2 in all_hydrogens) and (at1 in a_raw1):
            h_raw1.append(at2)
            d_raw1.append(at1)
        else:
            continue

    # Process the indices of donors to a numba dict
    d_raw = defaultdict(list)
    [d_raw[atoms_to_resids[x]].append(x) for x in d_raw1]
    d_raw3 = {x: np.asarray(d_raw[x], dtype=np.int32) for x in d_raw}
    donors = pydict_to_numbadict(d_raw3)

    # Process the indices of hydrogens to a numba dict
    h_raw = defaultdict(list)
    [h_raw[atoms_to_resids[x]].append(x) for x in h_raw1]
    h_raw3 = {x: np.asarray(h_raw[x], dtype=np.int32) for x in h_raw}
    hydros = pydict_to_numbadict(h_raw3)

    # Process the indices of acceptors to a numba dict
    a_raw = defaultdict(list)
    [a_raw[atoms_to_resids[x]].append(x) for x in a_raw1]
    a_raw3 = {x: np.asarray(a_raw[x], dtype=np.int32) for x in a_raw}
    acceptors = pydict_to_numbadict(a_raw3)
    return donors, hydros, acceptors


@njit(parallel=False)
def dict_get(dico, key):
    """
    Get the value of a key in a dictionary or return None if the key is not in
    the dictionary

    Args:
        dico: dictionary to search
        key: key to search

    Returns:
        value: value of the key in the dictionary or None if the key is not in
               the dictionary
    """
    try:
        value = dico[key]
        return value
    except:
        return None


def pydict_to_numbadict(py_dict):
    """
    Converts from Python dict to Numba dict

    Args:
        py_dict: Python dict

    Returns:
        numba_dict: Numba dict
    """
    numba_dict = Dict()
    for key in py_dict:
        numba_dict.update({key: py_dict[key]})
    return numba_dict


def to_matrix(one_dim_array, n, nested=False):
    """
    Converts a one-dimensional array of size = N * (N-1) / 2, into the
    equivalent N * N matrix

    Args:
        one_dim_array: one-dimensional array
        n: number of col / row in the square matrix

    Returns:
        matrix: N * N symmetrycal matrix
    """
    matrix = np.zeros((n, n))
    k = 0

    if nested:
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i, j] = one_dim_array[k][0]
                k += 1
    else:
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i, j] = one_dim_array[k]
                k += 1

    matrix += matrix.T
    return matrix


class Mapping:
    # todo: simiplify with prody or another pdb parser as explicit line
    #  handling in PDB can be tricky even if standard format exists
    """
    @Sneha's class for residue renumbering and mapping operations on PDB files.
    """

    def __init__(self, out_dir):
        self.out_dir = out_dir

    def pre_processing(self, input_pdb):
        # todo: handle topology formats others than PDB
        """
        Renumbers the residues in a PDB file sequentially from 1, changing all chain identifiers to 'A'.
        Outputs a renumbered PDB file and a map file that records the original and new residue numbers and chains.

        Parameters:
        input_pdb (str): Path to the input PDB file.
        """
        # Define file paths for the renumbered PDB file and the map file
        renumbered_pdb_raw = input_pdb.replace(".pdb", "_renumbered.pdb")
        renumbered_pdb = join(self.out_dir, basename(renumbered_pdb_raw))
        map_file_raw = input_pdb.replace(".pdb", "_map.txt")
        map_file = join(self.out_dir, basename(map_file_raw))

        # Open input PDB file for reading, renumbered PDB file and map file for writing
        with open(input_pdb, "r") as infile, open(renumbered_pdb,
                                                  "w") as outfile, open(
            map_file, "w"
        ) as mapfile:
            # Initialize variables for residue renumbering and mapping
            current_residue_number = 0
            residue_map = {}
            last_residue_id = None

            # Loop through each line in the input PDB file
            for line in infile:
                if line.startswith(("ATOM", "HETATM")):
                    # Extract chain ID, old residue number, and residue name
                    chain_id = line[21]
                    old_residue_number = line[22:26].strip()
                    res_name = line[17:20].strip()
                    residue_id = (chain_id, old_residue_number, res_name)

                    # Check if it's a new residue
                    if residue_id != last_residue_id:
                        current_residue_number += 1
                        last_residue_id = residue_id
                        residue_map[(chain_id, old_residue_number)] = (
                            current_residue_number,
                            "A",
                        )

                    # Write renumbered line with new chain ID 'A'
                    new_line = (
                            line[:21]
                            + "A"
                            + str(current_residue_number).rjust(4)
                            + line[26:]
                    )
                    outfile.write(new_line)
                else:
                    # Write non-ATOM/HETATM lines as they are
                    outfile.write(line)

            # Write the residue map to the map file
            for (chain_id, old_number), (
                    new_number, new_chain) in residue_map.items():
                mapfile.write(
                    f"{chain_id} {old_number} {new_chain} {new_number}\n")

        # Print confirmation messages
        # print(f"Renumbered PDB file saved as: {renumbered_pdb}")
        # print(f"Residue mapping file saved as: {map_file}")
        return map_file, renumbered_pdb

    def post_processing(self, map_file, renumbered_pdb):
        """
        Restores the original residue numbering and chain identifiers in a renumbered PDB file using the map file.

        Parameters:
        map_file (str): Path to the map file containing the original and new residue numbers and chains.
        renumbered_pdb (str): Path to the renumbered PDB file.
        """
        # Define file path for the restored PDB file
        original_pdb = renumbered_pdb.replace("_renumbered.pdb",
                                              "_restored.pdb")

        # Initialize a dictionary to store the residue mapping information
        residue_map = {}

        # Read the map file and populate the residue mapping dictionary
        with open(map_file, "r") as mapfile:
            for line in mapfile:
                original_chain, old_number, new_chain, new_number = line.split()
                residue_map[(new_chain, new_number)] = (
                    original_chain, old_number)

        # Open renumbered PDB file for reading and restored PDB file for writing
        with open(renumbered_pdb, "r") as infile, open(original_pdb,
                                                       "w") as outfile:
            # Loop through each line in the renumbered PDB file
            for line in infile:
                if line.startswith(("ATOM", "HETATM")):
                    # Extract new chain ID and new residue number
                    new_chain = line[21]
                    new_number = line[22:26].strip()
                    original_chain, old_number = residue_map[
                        (new_chain, new_number)]

                    # Write restored line with original chain ID and residue number
                    new_line = (
                            line[:21] + original_chain + old_number.rjust(
                        4) + line[26:]
                    )
                    outfile.write(new_line)
                else:
                    # Write non-ATOM/HETATM lines as they are
                    outfile.write(line)

        # Print confirmation message
        # print(f"Restored PDB file saved as: {original_pdb}")

# =============================================================================
#
# =============================================================================
# import mdtraj as md
# import prody as prd
#
# load topology and trajectory
# topo = '/home/rglez/RoyHub/compass/data/MDs/nucleosome_full_2c/1kx5_dry.pdb'
# traj = '/home/rglez/RoyHub/compass/data/MDs/nucleosome_full_2c/nuc-prot-trim.dcd'
# out_dir = '/home/rglez/RoyHub/compass/data/outputs/nucleosome_full_2c'

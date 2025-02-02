# Created by gonzalezroy at 7/27/24
from collections import defaultdict

import mdtraj as md
import networkx as nx
import numpy as np
from numba.typed import List

import topo_traj as tt


def get_cycles(bonds):
    """
    Find all cycles in a topology.

    Args:
        bonds: List of bonds in the topology.
        max_size: Maximum size of the cycles to find.

    Returns:
        List of cycles in the topology.
    """

    # Create a dictionary of bonds for each atom
    topo_bonds = defaultdict(list)
    for bond in bonds:
        atom1 = bond[0].index
        atom2 = bond[1].index
        topo_bonds[atom1].append(atom2)

    # Convert topo_bonds to a NetworkX graph
    G = nx.Graph(topo_bonds)
    all_cycles = list(nx.simple_cycles(G))

    # Find all cycles in the topo_bonds graph
    interval = range(5, 7)
    cycles = [x for x in all_cycles if len(x) in interval]

    return cycles


def calculate_dihedral_angle(coords1, coords2, coords3, coords4):
    """
    Calculate the dihedral angle between four points in space.

    Args:
        coords1, coords2, coords3, coords4: Coordinates of the four points.

    Returns:
        Dihedral angle in degrees.
    """
    b1 = coords2 - coords1
    b2 = coords3 - coords2
    b3 = coords4 - coords3

    b1xb2 = np.cross(b1, b2)
    b2xb3 = np.cross(b2, b3)

    b1xb2_x_b2xb3 = np.cross(b1xb2, b2xb3)

    y = np.dot(b1xb2_x_b2xb3, b2) * (1.0 / np.linalg.norm(b2))
    x = np.dot(b1xb2, b2xb3)

    return np.degrees(np.arctan2(y, x))


def is_planar_cycle(cycle, coordinates, tolerance=10):
    """
    Check if a given cycle is planar by calculating dihedral angles.

    Args:
        cycle: List of atom indices representing the cycle.
        coordinates: Array of coordinates for all atoms.
        tolerance: Tolerance in degrees for planarity.

    Returns:
        Boolean indicating if the cycle is planar.
    """
    num_atoms = len(cycle)
    for i in range(num_atoms):
        coords1 = coordinates[cycle[i]]
        coords2 = coordinates[cycle[(i + 1) % num_atoms]]
        coords3 = coordinates[cycle[(i + 2) % num_atoms]]
        coords4 = coordinates[cycle[(i + 3) % num_atoms]]

        dihedral_angle = calculate_dihedral_angle(coords1, coords2, coords3,
                                                  coords4)
        if abs(dihedral_angle) > tolerance and abs(
                dihedral_angle - 180) > tolerance:
            return False
    return True


# %%=============================================================================
# Debugging Area
# =============================================================================
import MDAnalysis as mda

topo = '/home/rglez/RoyHub/oxo-8/data/raw/A1/8oxoGA1_1_dry.prmtop'
traj = '/home/rglez/RoyHub/oxo-8/data/raw/A1/8oxoGA1_1_dry.nc'

topo = '/home/rglez/RoyHub/oxo-8/data/raw/water/A1/8oxoGA1_1_hmr.prmtop'
traj = '/home/rglez/RoyHub/oxo-8/data/raw/water/A1/8oxoGA1_1_sk100.nc'

sel1 = "nucleic or resname 8OG"
sel2 = 'resname WAT'

u = mda.Universe(topo, traj, to_guess=['types'], force_guess=True)

u_resids, u_indices = np.unique(u.residues.resnames, return_index=True)
for uinque_idx in u_indices:
    s = u.select_atoms(f"resindex {uinque_idx}")
    mapp = s.atoms.indices
    try:
        mol = s.atoms.convert_to("RDKIT")

    except AttributeError:
        print(f"Failed to convert {s.residues.resnames[0]}")
        continue

    except Exception as e:
        s.guess_bonds()
        print('rdkit')


def isRingAromatic(mol, mapp):
    ri = mol.GetRingInfo()
    bond_rings = ri.BondRings()

    aromatics = []
    for bond_ring in bond_rings:

        count = 0
        atoms = set()
        for id in bond_ring:
            bond = mol.GetBondWithIdx(id)
            at1 = bond.GetBeginAtomIdx()
            at2 = bond.GetEndAtomIdx()
            atoms.add(at1)
            atoms.add(at2)

            if not bond.GetIsAromatic():
                count += 1

        if count == 0:
            aromatics.append(mapp[list(atoms)])

    return aromatics


aromatics = isRingAromatic(mol, mapp)

mini_traj = next(md.iterload(traj, top=topo, chunk=1))
full_topo = mini_traj.topology.to_dataframe()[0]

# Indices of residues in the load trajectory and equivalence
resids_to_atoms, resids_to_noh, internal_equiv = tt.get_resids_indices(
    mini_traj)
raw = {y: x for x in resids_to_atoms for y in resids_to_atoms[x]}
atoms_to_resids = tt.pydict_to_numbadict(raw)

# Get the planar cycles
bonds = mini_traj.topology.bonds
cycles_raw = get_cycles(bonds)
coordinates = mini_traj.xyz[0]
cycles = [
    cycle for cycle in cycles_raw if
    is_planar_cycle(cycle, coordinates)]

resids_to_rings_raw = defaultdict(List)
for cycle in cycles:
    ring_resid = atoms_to_resids[cycle[0]]
    resids_to_rings_raw[ring_resid].append(List(cycle))
resids_to_rings = tt.pydict_to_numbadict(resids_to_rings_raw)

# =============================================================================
#
# =============================================================================
from numba import njit, prange


@njit
def calc_normal_vector(p1, p2, p3):
    # Calculate vectors from points
    v1 = p2 - p1
    v2 = p3 - p1

    # Calculate the normal vector
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)

    if norm == 0:
        return np.zeros(3)

    return normal / norm


@njit
def calc_ring_center(ring):
    center = np.zeros(3)
    for atom in ring:
        center += atom
    return center / len(ring)


@njit(parallel=True)
def detect_stacking(rings):
    num_rings = len(rings)
    interactions = []

    for i in prange(num_rings):
        for j in range(i + 1, num_rings):
            ring1 = rings[i]
            ring2 = rings[j]

            # Compute normal vectors for both rings
            normal1 = calc_normal_vector(ring1[0], ring1[1], ring1[2])
            normal2 = calc_normal_vector(ring2[0], ring2[1], ring2[2])

            # Compute centers of the rings
            center1 = calc_ring_center(ring1)
            center2 = calc_ring_center(ring2)

            # Calculate distance between ring centers
            center_dist = np.linalg.norm(center2 - center1)

            # Calculate angle between normal vectors
            dot_product = np.dot(normal1, normal2)
            angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
            angle_degrees = np.degrees(angle)

            # Determine if stacking interaction exists
            if center_dist < 5.0 and (
                    abs(angle_degrees) < 30 or abs(angle_degrees - 180) < 30
            ):
                interactions.append((i, j))

    return interactions

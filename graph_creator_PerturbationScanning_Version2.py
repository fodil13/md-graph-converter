
"""
Molecular Dynamics Graph Converter
Copyright (c) 2025 Fodil Azzaz, PhD - All Rights Reserved
Non-commercial use only
Commercial? Contact me: azzaz.fodil@gmail.com
Citation: https://doi.org/10.64898/2025.12.09.692808

Converts MD simulation frames into graphs with 13D node features,
non-covalent edges, and SASA-based interface stability score.
"""
import os
import glob
import numpy as np
from scipy.spatial import cKDTree
import MDAnalysis as mda
import torch
from torch_geometric.data import Data
from google.colab import files
from google.colab import drive
import warnings
import tempfile
import freesasa

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== CONSTANTS ==========
LIPID_RESNAMES = ['POPC', 'CHL1', 'ANE5AC', 'CER160', 'BGLC', 'BGAL', 'BGALNA', 'POPE', 'POPS', 'CHOL']
WATER_RESNAMES = ['TIP3', 'SOL', 'WAT', 'HOH', 'TIP4P']
AMINO_ACIDS = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
               'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
SEGIDS = ['PROA', 'PROB', 'GLIP', 'GLIZ', 'POPC', 'CHL1']

# ========== PSF PARSING ==========
def parse_psf_charges_masses(psf_filename):
    charges, masses = {}, {}
    atoms_parsed = 0
    try:
        with open(psf_filename, 'r') as f:
            lines = f.readlines()
        in_atoms = False
        for line in lines:
            if '!NATOM' in line:
                in_atoms = True
                print(" Found ATOM section in PSF")
                continue
            elif in_atoms and ('!NBOND' in line or not line.strip()):
                in_atoms = False
                break
            if in_atoms and line.strip() and not line.startswith('!'):
                parts = line.split()
                if len(parts) >= 8:
                    resname = parts[3]
                    atom_name = parts[4]
                    charge_str = parts[6]
                    mass_str = parts[7]
                    key = f"{resname}_{atom_name}"
                    try:
                        charges[key] = float(charge_str)
                        masses[key] = float(mass_str)
                        atoms_parsed += 1
                        if atoms_parsed <= 3:
                            print(f"    Sample: {key} → charge={charges[key]:.3f}, mass={masses[key]:.1f}")
                    except ValueError:
                        continue
        print(f" PSF parsed: {atoms_parsed} atom charges/masses")
        if atoms_parsed == 0:
            charges, masses = create_fallback_charges_masses()
    except Exception as e:
        print(f" PSF parsing error: {e}")
        charges, masses = create_fallback_charges_masses()
    return charges, masses

def create_fallback_charges_masses():
    print(" Creating fallback charges...")
    charges, masses = {}, {}
    atom_charges = {
        'N': -0.47, 'CA': 0.07, 'C': 0.51, 'O': -0.51, 'OXT': -0.51,
        'H': 0.31, 'HA': 0.09, 'HB': 0.09, 'HG': 0.09, 'HD': 0.09, 'HE': 0.09,
        'CB': 0.05, 'CG': -0.08, 'CD': -0.18, 'CE': -0.30, 'CZ': 0.25,
        'OG': -0.66, 'OG1': -0.66, 'OD1': -0.76, 'OD2': -0.76, 'OE1': -0.76, 'OE2': -0.76,
        'ND1': -0.40, 'ND2': -0.60, 'NE': -0.70, 'NE1': -0.70, 'NE2': -0.70, 'NZ': -0.80,
        'SG': -0.16, 'SD': 0.45, 'P': 1.50,
        'C': 0.00, 'O': -0.50, 'N': -0.30, 'H': 0.30
    }
    atom_masses = {'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'S': 32.06, 'P': 30.974}
    all_residues = AMINO_ACIDS + LIPID_RESNAMES + WATER_RESNAMES
    for atom_type, charge in atom_charges.items():
        for resname in all_residues:
            key = f"{resname}_{atom_type}"
            charges[key] = charge
            if atom_type.startswith('H'):
                masses[key] = atom_masses['H']
            elif atom_type.startswith('C'):
                masses[key] = atom_masses['C']
            elif atom_type.startswith('N'):
                masses[key] = atom_masses['N']
            elif atom_type.startswith('O'):
                masses[key] = atom_masses['O']
            elif atom_type.startswith('S'):
                masses[key] = atom_masses['S']
            elif atom_type.startswith('P'):
                masses[key] = atom_masses['P']
            else:
                masses[key] = 12.011
    print(f" Created fallback charges for {len(charges)} atom types")
    return charges, masses

def parse_psf_bonds(psf_filename):
    bonds = []
    try:
        with open(psf_filename, 'r') as f:
            lines = f.readlines()
        in_bonds = False
        for line in lines:
            if '!NBOND' in line:
                in_bonds = True
                continue
            if in_bonds:
                if not line.strip() or '!NTHETA' in line:
                    break
                if line.strip().startswith('!'):
                    continue
                parts = line.split()
                for i in range(0, len(parts), 2):
                    if i+1 < len(parts):
                        try:
                            a1 = int(parts[i]) - 1
                            a2 = int(parts[i+1]) - 1
                            bonds.append((a1, a2))
                        except ValueError:
                            continue
        print(f" {len(bonds)} covalent bonds parsed from PSF")
    except Exception as e:
        print(f" Bond parsing failed: {e}")
    return bonds

# ========== FEATURE ENGINEERING ==========
def get_element_from_atom_name(atom_name, resname):
    clean_name = ''.join([c for c in atom_name if not c.isdigit()]).strip()
    element_map = {
        'N': 'N', 'CA': 'C', 'C': 'C', 'O': 'O', 'OXT': 'O',
        'CB': 'C', 'CG': 'C', 'CD': 'C', 'CE': 'C', 'CZ': 'C',
        'CG1': 'C', 'CG2': 'C', 'CD1': 'C', 'CD2': 'C',
        'OG': 'O', 'OG1': 'O', 'OD1': 'O', 'OD2': 'O', 'OE1': 'O', 'OE2': 'O',
        'SD': 'S', 'SG': 'S', 'H': 'H', 'HA': 'H', 'HB': 'H', 'HG': 'H', 'HD': 'H',
        'P': 'P', 'OW': 'O', 'HW1': 'H', 'HW2': 'H'
    }
    if clean_name in element_map:
        return element_map[clean_name]
    if clean_name.startswith('H'): return 'H'
    elif clean_name.startswith('C'): return 'C'
    elif clean_name.startswith('N'): return 'N'
    elif clean_name.startswith('O'): return 'O'
    elif clean_name.startswith('S'): return 'S'
    elif clean_name.startswith('P'): return 'P'
    return 'C'

def get_atomic_number_from_element(element):
    element_to_number = {'H': 1.0, 'C': 6.0, 'N': 7.0, 'O': 8.0, 'S': 16.0, 'P': 15.0}
    return element_to_number.get(element, 6.0)

def get_residue_hydrophobicity(resname):
    hydrophobicity_scale = {
        'ALA': 1.8, 'VAL': 4.2, 'LEU': 3.8, 'ILE': 4.5, 'PHE': 2.8, 'TRP': -0.9,
        'MET': 1.9, 'PRO': -1.6, 'GLY': -0.4, 'SER': -0.8, 'THR': -0.7, 'CYS': 2.5,
        'TYR': -1.3, 'ASN': -3.5, 'GLN': -3.5, 'ASP': -3.5, 'GLU': -3.5,
        'LYS': -3.9, 'ARG': -4.5, 'HIS': -3.2
    }
    raw = hydrophobicity_scale.get(resname, 0.0)
    return max(-1.0, min(1.0, raw / 4.5))

def get_residue_charge_category(resname):
    positive = ['LYS', 'ARG', 'HIS', 'HSD', 'HSE', 'HSP']
    negative = ['ASP', 'GLU']
    if resname in positive: return 1.0
    if resname in negative: return -1.0
    return 0.0

def create_equiformerv2_features(atom, psf_charges, psf_masses):
    key = f"{atom.resname}_{atom.name}"
    charge = psf_charges.get(key, 0.0)
    mass = psf_masses.get(key, 12.0) / 100.0
    element = get_element_from_atom_name(atom.name, atom.resname)
    atomic_number = get_atomic_number_from_element(element) / 10.0
    residue_hydrophobicity = get_residue_hydrophobicity(atom.resname)
    residue_charge_category = get_residue_charge_category(atom.resname)
    is_backbone = 1.0 if atom.name in ['N', 'CA', 'C', 'O', 'OXT'] else 0.0
    is_sidechain = 1.0 if atom.name not in ['N', 'CA', 'C', 'O', 'OXT'] and atom.resname in AMINO_ACIDS else 0.0
    segid = getattr(atom, 'segid', 'other')
    segid_features = [1.0 if segid == seg else 0.0 for seg in SEGIDS]
    features = [charge, mass, atomic_number, residue_hydrophobicity,
                residue_charge_category, is_backbone, is_sidechain, *segid_features]
    return features

def radial_basis_functions(distance, num_rbf=16, rbf_min=0.0, rbf_max=20.0):
    centers = np.linspace(rbf_min, rbf_max, num_rbf)
    width = (rbf_max - rbf_min) / num_rbf
    rbf_values = np.exp(-((distance - centers) ** 2) / (2 * width ** 2))
    return rbf_values.tolist()

# ========== EDGE CREATION ==========
def create_non_covalent_edges(positions, segids, max_distance=6.0):
    print(f"     Creating NON-COVALENT edges (≤ {max_distance}Å)...")
    non_covalent_edges = []
    positions_np = np.array(positions)
    pair_counts = {
        'PROA_GLIZ': 0, 'GLIZ_PROA': 0, 'PROB_GLIZ': 0, 'GLIZ_PROB': 0,
        'PROA_PROB': 0, 'PROB_PROA': 0, 'PROA_GLIP': 0, 'GLIP_PROA': 0,
        'PROB_GLIP': 0, 'GLIP_PROB': 0, 'GLIP_GLIZ': 0, 'GLIZ_GLIP': 0,
        'OTHER': 0
    }
    tree = cKDTree(positions_np)
    pairs = tree.query_pairs(max_distance)
    for i, j in pairs:
        segid_i, segid_j = segids[i], segids[j]
        if segid_i != segid_j:
            pair_key = f"{segid_i}_{segid_j}"
            reverse_key = f"{segid_j}_{segid_i}"
            if pair_key in pair_counts:
                pair_counts[pair_key] += 1
            elif reverse_key in pair_counts:
                pair_counts[reverse_key] += 1
            else:
                pair_counts['OTHER'] += 1
            non_covalent_edges.append([min(i, j), max(i, j)])
    total = len(non_covalent_edges)
    for pt, cnt in pair_counts.items():
        if cnt > 0:
            print(f"       {pt}: {cnt} edges ({cnt/total*100:.1f}%)")
    print(f"     {total} non-covalent edges total")
    return non_covalent_edges

def create_edges_with_rbf(positions, psf_bonds, atom_index_to_node_index, segids, cutoff=6.0):
    print(f"    🔗 Creating edges for {len(positions)} atoms...")
    covalent_edges = []
    covalent_pairs = set()
    for a1, a2 in psf_bonds:
        if a1 in atom_index_to_node_index and a2 in atom_index_to_node_index:
            n1 = atom_index_to_node_index[a1]
            n2 = atom_index_to_node_index[a2]
            if n1 != n2:
                covalent_edges.append([min(n1, n2), max(n1, n2)])
                covalent_pairs.add((min(n1, n2), max(n1, n2)))
    print(f"     {len(covalent_edges)} covalent bonds")
    non_covalent_edges = create_non_covalent_edges(positions, segids, max_distance=cutoff)
    all_edges_undirected = covalent_edges + non_covalent_edges
    all_edges_pyg = []
    for src, tgt in all_edges_undirected:
        all_edges_pyg.append([src, tgt])
        all_edges_pyg.append([tgt, src])
    total_cov = len(covalent_edges)
    total_non = len(non_covalent_edges)
    total_pyg = len(all_edges_pyg) // 2
    print(f"     EDGE SUMMARY: Covalent={total_cov}, Non-covalent={total_non}, TOTAL={total_pyg}")
    return all_edges_pyg, covalent_pairs

def create_edge_features(positions, edges, covalent_pairs):
    edge_scalar = []
    edge_vector = []
    for src, tgt in edges:
        diff = positions[tgt] - positions[src]
        dist = np.linalg.norm(diff)
        rbf = radial_basis_functions(dist)
        is_cov = 1.0 if (min(src, tgt), max(src, tgt)) in covalent_pairs else 0.0
        edge_scalar.append([*rbf, is_cov])
        if dist > 1e-8:
            edge_vector.append(diff / dist)
        else:
            edge_vector.append([1.0, 0.0, 0.0])
    return edge_scalar, edge_vector

# ========== SASA FUNCTIONS (CORRECTED) ==========
def compute_buried_sasa(positions, elements, segids, segA, segB, probe=1.4):
    """
    Compute buried surface area between two groups using freesasa.
    Returns area in Å², or 0.0 if fails.
    """
    # Create temporary PDB file with only atoms from segA and segB
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
        pdb_filename = tmp.name
        for i, (pos, elem, seg) in enumerate(zip(positions, elements, segids), start=1):
            if seg not in (segA, segB):
                continue
            # PDB fixed format
            atom_name = elem.ljust(4)
            res_name = 'XXX'
            chain = seg[0] if seg else 'A'
            res_seq = i % 10000
            x, y, z = pos
            line = f"ATOM  {i:5d} {atom_name:4s} {res_name:3s} {chain:1s}{res_seq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:2s}\n"
            tmp.write(line)
        tmp.flush()
    try:
        structure = freesasa.Structure(pdb_filename)
        result = freesasa.calc(structure)
        total_sasa = result.totalArea()

        # Compute SASA for each segment separately by creating two temporary PDBs
        def sasa_for_seg(seg):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as seg_tmp:
                seg_filename = seg_tmp.name
                for i, (pos, elem, s) in enumerate(zip(positions, elements, segids), start=1):
                    if s != seg:
                        continue
                    atom_name = elem.ljust(4)
                    res_name = 'XXX'
                    chain = s[0] if s else 'A'
                    res_seq = i % 10000
                    x, y, z = pos
                    line = f"ATOM  {i:5d} {atom_name:4s} {res_name:3s} {chain:1s}{res_seq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:2s}\n"
                    seg_tmp.write(line)
                seg_tmp.flush()
            try:
                s = freesasa.Structure(seg_filename)
                r = freesasa.calc(s)
                return r.totalArea()
            finally:
                os.unlink(seg_filename)

        sasaA = sasa_for_seg(segA)
        sasaB = sasa_for_seg(segB)
        buried = (sasaA + sasaB - total_sasa) / 2.0
        return max(0.0, buried)
    except Exception as e:
        # Silent fail – will be caught by caller
        return 0.0
    finally:
        os.unlink(pdb_filename)

def compute_contact_count_fallback(positions, elements, segids, segA, segB, cutoff=4.5):
    """Fallback: count heavy atom contacts (< cutoff) between segments."""
    idxA = [i for i, s in enumerate(segids) if s == segA]
    idxB = [i for i, s in enumerate(segids) if s == segB]
    if not idxA or not idxB:
        return 0.0
    # Keep only heavy atoms (not hydrogen)
    heavy_A = [i for i in idxA if elements[i] != 'H']
    heavy_B = [i for i in idxB if elements[i] != 'H']
    if not heavy_A or not heavy_B:
        return 0.0
    posA = np.array([positions[i] for i in heavy_A])
    posB = np.array([positions[i] for i in heavy_B])
    tree = cKDTree(posA)
    distances, _ = tree.query(posB, k=1)
    return float(np.sum(distances < cutoff))

# ========== GRAPH CREATION ==========
def create_equiformerv2_graph_from_frame(frame_idx, universe, psf_charges, psf_masses, psf_bonds,
                                       protein_selection='protein', environment_selection=None):
    print(f"   Processing frame {frame_idx}...")
    universe.trajectory[frame_idx]

    protein_atoms = universe.select_atoms(protein_selection)
    if environment_selection:
        environment_atoms = universe.select_atoms(environment_selection)
    else:
        environment_atoms = universe.select_atoms('not protein')
    all_atoms = protein_atoms + environment_atoms
    if len(all_atoms) == 0:
        print(f" No atoms found!")
        return None
    print(f"    • Total atoms: {len(all_atoms)}")
    print(f"    • Protein atoms: {len(protein_atoms)}")
    print(f"    • Environment atoms: {len(environment_atoms)}")

    all_atom_indices = set(atom.index for atom in all_atoms)
    valid_psf_bonds = []
    fully_inside_bonds = 0
    boundary_bonds = 0
    for a1, a2 in psf_bonds:
        a1_in = a1 in all_atom_indices
        a2_in = a2 in all_atom_indices
        if a1_in and a2_in:
            valid_psf_bonds.append((a1, a2))
            fully_inside_bonds += 1
        elif a1_in or a2_in:
            valid_psf_bonds.append((a1, a2))
            boundary_bonds += 1
    print(f"     Bonds: {fully_inside_bonds} inside, {boundary_bonds} boundary")

    atom_positions = []
    atom_features = []
    atom_index_to_node_index = {}
    all_atom_segids = []
    all_atom_resnames = []
    all_atom_residues = []
    atom_elements = []
    charge_values = []

    for node_idx, atom in enumerate(all_atoms):
        atom_positions.append(atom.position.copy())
        features = create_equiformerv2_features(atom, psf_charges, psf_masses)
        atom_features.append(features)
        atom_index_to_node_index[atom.index] = node_idx
        seg = getattr(atom, 'segid', 'UNK')
        all_atom_segids.append(seg)
        all_atom_resnames.append(atom.resname)
        all_atom_residues.append(atom.resid)
        charge_values.append(features[0])
        elem = get_element_from_atom_name(atom.name, atom.resname)
        atom_elements.append(elem)

    print(f"     Processed {len(atom_features)} atoms, feature dim {len(atom_features[0])}")
    print(f"     SEGIDs: {set(all_atom_segids)}")

    # Create edges
    edge_index, covalent_pairs = create_edges_with_rbf(atom_positions, valid_psf_bonds,
                                                       atom_index_to_node_index, all_atom_segids)
    edge_scalar, edge_vector = create_edge_features(atom_positions, edge_index, covalent_pairs)

    # Tensors
    scalar_tensor = torch.FloatTensor(np.array(atom_features)).to(device)
    pos_tensor = torch.FloatTensor(np.array(atom_positions)).to(device)
    edge_index_tensor = torch.LongTensor(edge_index).t().contiguous().to(device)
    edge_scalar_tensor = torch.FloatTensor(np.array(edge_scalar)).to(device)
    edge_vector_tensor = torch.FloatTensor(np.array(edge_vector)).to(device).unsqueeze(1)

    graph = Data(
        x=scalar_tensor,
        pos=pos_tensor,
        edge_index=edge_index_tensor,
        edge_attr=(edge_scalar_tensor, edge_vector_tensor),
        original_positions=pos_tensor.clone(),
        num_atoms=len(atom_features),
        num_edges=edge_index_tensor.shape[1] // 2,
        frame_idx=frame_idx,
        time_ps=universe.trajectory.time,
        segids=all_atom_segids,
        resnames=all_atom_resnames,
        residues=all_atom_residues,
        num_boundary_bonds=boundary_bonds,
    )

    # Compute SASA score (sum of buried areas between all distinct segid pairs)
    unique_segids = sorted(set(all_atom_segids))
    total_buried = 0.0
    for i in range(len(unique_segids)):
        for j in range(i+1, len(unique_segids)):
            buried = compute_buried_sasa(atom_positions, atom_elements, all_atom_segids,
                                         unique_segids[i], unique_segids[j])
            if buried == 0.0:
                # Fallback to contact count
                buried = compute_contact_count_fallback(atom_positions, atom_elements, all_atom_segids,
                                                        unique_segids[i], unique_segids[j]) * 0.5  # scale to approx Å²
            total_buried += buried
    graph.sasa_score = total_buried
    print(f"     SASA score (total buried area): {total_buried:.2f} Å²")

    return graph

# ========== FILE HANDLING ==========
def setup_google_drive_files(psf_filename, dcd_filename):
    print("\n📁 SETTING UP GOOGLE DRIVE FILES...")
    drive.mount('/content/drive')
    psf_basename = os.path.basename(psf_filename) if psf_filename else None
    dcd_basename = os.path.basename(dcd_filename) if dcd_filename else None
    psf_path, dcd_path = None, None
    for root, dirs, files in os.walk('/content/drive/MyDrive'):
        if psf_basename in files:
            psf_path = os.path.join(root, psf_basename)
        if dcd_basename in files:
            dcd_path = os.path.join(root, dcd_basename)
    if not psf_path or not dcd_path:
        print("❌ Files not found in Google Drive!")
        return None, None
    import shutil
    shutil.copy(psf_path, psf_basename)
    shutil.copy(dcd_path, dcd_basename)
    print(f"✅ Files copied: {psf_basename}, {dcd_basename}")
    return psf_basename, dcd_basename

def setup_md_files(psf_filename=None, dcd_filename=None):
    if psf_filename or dcd_filename:
        return setup_google_drive_files(psf_filename, dcd_filename)
    psf_files = glob.glob("*.psf")
    dcd_files = glob.glob("*.dcd")
    if psf_files and dcd_files:
        return psf_files[0], dcd_files[0]
    print("\n UPLOAD MD FILES")
    uploaded = files.upload()
    psf_file = next((f for f in uploaded if f.endswith('.psf')), None)
    dcd_file = next((f for f in uploaded if f.endswith('.dcd')), None)
    if not psf_file or not dcd_file:
        print("❌ Missing PSF or DCD file!")
        return None, None
    return psf_file, dcd_file

def run_equiformerv2_pipeline(psf_filename=None, dcd_filename=None,
                            protein_selection='protein', environment_selection=None,
                            num_frames=3, frame_step=1):
    print(" EQUIFORMERV2 GENERATIVE GRAPH PIPELINE (with SASA)")
    print(f" Protein: {protein_selection}")
    print(f" Environment: {environment_selection if environment_selection else 'all non-protein'}")
    psf_file, dcd_file = setup_md_files(psf_filename, dcd_filename)
    if not psf_file or not dcd_file:
        return None
    print("\n  PARSING PSF FILE...")
    psf_charges, psf_masses = parse_psf_charges_masses(psf_file)
    psf_bonds = parse_psf_bonds(psf_file)
    u = mda.Universe(psf_file, dcd_file)
    print(f"✅ System loaded: {len(u.atoms)} atoms, {len(u.trajectory)} frames")
    graphs = []
    frame_indices = list(range(0, min(num_frames * frame_step, len(u.trajectory)), frame_step))
    for i, frame_idx in enumerate(frame_indices):
        graph = create_equiformerv2_graph_from_frame(
            frame_idx, u, psf_charges, psf_masses, psf_bonds,
            protein_selection=protein_selection,
            environment_selection=environment_selection
        )
        if graph is not None:
            graphs.append(graph)
            print(f"    ✅ Frame {frame_idx} completed ({i+1}/{len(frame_indices)})")
    if graphs:
        filename = f"equiformerv2_graphs_{len(graphs)}frames_WITH_SASA.pt"
        torch.save(graphs, filename)
        print(f" Saved {len(graphs)} graphs as {filename}")
        try:
            files.download(filename)
            print(f"✅ File downloaded: {filename}")
        except:
            print(f"📁 File saved locally: {filename}")
    return graphs

# === EXECUTION ===
if __name__ == "__main__":
    PSF_FILENAME = "/content/drive/MyDrive/ionized.psf"
    DCD_FILENAME = "/content/drive/MyDrive/KV_CLR_full_movie.dcd"
    graphs = run_equiformerv2_pipeline(
        psf_filename=PSF_FILENAME,
        dcd_filename=DCD_FILENAME,
        protein_selection='segid PROA',
        environment_selection='byres (segid MEMB and around 3 segid PROA)',
        num_frames=10000,
        frame_step=5
    )

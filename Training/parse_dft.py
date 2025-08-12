#!/usr/bin/env python3
import os
import re
import warnings
import logging
import pickle
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from ase import Atoms
from ase.io import read, write
from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.singlepoint import SinglePointCalculator
from pymatgen.io.ase import AseAtomsAdaptor

# ---------------------------
# Constants / Unit conversion
# ---------------------------
EV_PER_HARTREE = 27.211386245988
ANG_PER_BOHR = 0.529177210903
HARTREE_PER_BOHR_TO_EV_PER_A = EV_PER_HARTREE / ANG_PER_BOHR  # ≈ 51.4220674763


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Parse VASP/ORCA directories and process atomic data")
    parser.add_argument('--input_filepaths', nargs='+', required=True, help="Paths to input directories")
    parser.add_argument('--output', required=True, help="Output file (pickle or XYZ prefix for --mace)")
    parser.add_argument('--filter', action='store_true', help="Apply outlier filtering")
    parser.add_argument('--graph', action='store_true', help="Generate graphs")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose output")
    parser.add_argument('--mace', action='store_true', help="Prepare MACE-compatible data (XYZ split)")
    parser.add_argument('--stepsize', type=int, default=1, help="Sampling step size")
    parser.add_argument('--orca', action='store_true', help="Parse ORCA .out files (last step only)")
    parser.add_argument('--orca-all-steps', action='store_true', help="Parse all ORCA optimization steps")
    return parser.parse_args()


# ---------------------------
# VASP/QE parsing (existing)
# ---------------------------
def parse_vasp_dir(filepaths, verbose, stepsize=1):
    """Parse multiple VASP/QE directories and extract relevant atomic structures."""
    warnings.filterwarnings('ignore')
    atoms_list = []

    total_iterations = sum(len(os.listdir(fp)) for fp in filepaths)

    with tqdm(total=total_iterations, desc='Processing VASP/QE') as pbar:
        for filepath in filepaths:
            for i in os.listdir(filepath):
                outcar_path = Path(filepath) / i / 'OUTCAR'
                pwscf_path = Path(filepath) / i / 'pwscf.out'

                if outcar_path.exists():
                    file_path = outcar_path
                elif pwscf_path.exists():
                    file_path = pwscf_path
                else:
                    if verbose:
                        print(f"No valid file ('OUTCAR' or 'pwscf.out') found in {Path(filepath) / i}")
                    pbar.update(1)
                    continue

                try:
                    single_file_atom = read(file_path, index=':')
                    # Sample atoms based on stepsize
                    for a in list(single_file_atom)[::stepsize]:
                        if a.get_total_energy() < 0:
                            a.info['file'] = str(Path(filepath) / i)
                            atoms_list.append(a)
                except Exception as e:
                    if verbose:
                        print(f"Error reading file: {file_path}")
                        print(f"Error details: {e}")
                finally:
                    pbar.update(1)

    return atoms_list


# ---------------------------
# ORCA parsing helpers
# ---------------------------
_ORCA_COORD_BLOCK_RE = re.compile(
    r"CARTESIAN COORDINATES\s+\(ANGSTROEM\)\s*\n-+\n((?:.*?\n)+?)-+\n", re.DOTALL
)
_ORCA_GRAD_BLOCK_RE = re.compile(
    r"CARTESIAN GRADIENT\s*\n-+\n((?:.*?\n)+?)-+\n", re.DOTALL
)
# Only accept lines that begin with index + element + ":" to avoid invariance diagnostics
_ORCA_GRAD_LINE_RE = re.compile(
    r"^\s*(\d+)\s+[A-Za-z]{1,3}\s*:\s*"
    r"([+-]?\d*\.?\d+(?:[EeDd][+-]?\d+)?)\s+"
    r"([+-]?\d*\.?\d+(?:[EeDd][+-]?\d+)?)\s+"
    r"([+-]?\d*\.?\d+(?:[EeDd][+-]?\d+)?)\s*$"
)

def _orca_parse_all_coords(content: str):
    """Return list of (symbols, positions[n,3]) for all coordinate blocks (Å)."""
    blocks = _ORCA_COORD_BLOCK_RE.findall(content)
    result = []
    for b in blocks:
        symbols, pos = [], []
        for line in b.strip().splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            sym = parts[0]
            try:
                x, y, z = [float(t.replace('D','E')) for t in parts[-3:]]
            except ValueError:
                continue
            symbols.append(sym)
            pos.append([x, y, z])
        if pos:
            result.append((symbols, np.array(pos, dtype=float)))
    return result

def _orca_parse_all_gradients(content: str, to_ev_per_A=True):
    """Return list of forces arrays (n,3) for each gradient block."""
    blocks = _ORCA_GRAD_BLOCK_RE.findall(content)
    out = []
    for b in blocks:
        rows = []
        for line in b.strip().splitlines():
            m = _ORCA_GRAD_LINE_RE.match(line)
            if not m:
                continue
            dEdx = float(m.group(2).replace('D','E'))
            dEdy = float(m.group(3).replace('D','E'))
            dEdz = float(m.group(4).replace('D','E'))
            fx, fy, fz = -dEdx, -dEdy, -dEdz  # force = -grad
            if to_ev_per_A:
                fx *= HARTREE_PER_BOHR_TO_EV_PER_A
                fy *= HARTREE_PER_BOHR_TO_EV_PER_A
                fz *= HARTREE_PER_BOHR_TO_EV_PER_A
            rows.append([fx, fy, fz])
        if rows:
            out.append(np.array(rows, dtype=float))
    return out

def _orca_parse_all_energies(content: str):
    """
    Collect energies per step if available.
    We try (in order):
      - 'TOTAL ENERGY' (appears per opt step)
      - 'FINAL SINGLE POINT ENERGY' (last step)
    Returns a list of floats (Hartree). May be shorter than coords/forces lists.
    """
    # TOTAL ENERGY ... -xxx.xxxxxx
    step_E = re.findall(r"TOTAL ENERGY\s+(-?\d+\.\d+(?:[Ee][+-]?\d+)?)", content)
    energies = [float(x) for x in step_E]
    # Final fallback (ensure last energy present)
    final_E = re.findall(
        r"FINAL SINGLE POINT ENERGY(?: \(From external program\))?\s+(-?\d+\.\d+(?:[Ee][+-]?\d+)?)",
        content
    )
    if final_E:
        e_last = float(final_E[-1])
        if not energies or (energies and abs(energies[-1] - e_last) > 1e-12):
            energies.append(e_last)
    return energies


# ---------------------------
# ORCA directory walkers
# ---------------------------
def parse_orca_dir_laststep(filepaths, verbose, stepsize=1):
    """
    Find *.out files, parse the LAST step (coords/energy/forces), attach SPC,
    return list of ASE Atoms.
    """
    atoms_list = []
    files = []
    for fp in filepaths:
        for root, _, fnames in os.walk(fp):
            for name in fnames:
                if name.lower().endswith('.out'):
                    files.append(Path(root) / name)

    with tqdm(total=len(files), desc='Processing ORCA (last step)') as pbar:
        for path in files[::stepsize]:
            try:
                content = Path(path).read_text(errors='ignore')
                coords_all = _orca_parse_all_coords(content)
                grads_all  = _orca_parse_all_gradients(content, to_ev_per_A=True)
                energies_all = _orca_parse_all_energies(content)

                if not coords_all:
                    if verbose:
                        print(f"[WARN] No coordinates found in {path}")
                    pbar.update(1)
                    continue

                symbols, positions = coords_all[-1]
                forces = grads_all[-1] if grads_all else None
                energy_h = energies_all[-1] if energies_all else None

                if (symbols is None) or (positions is None) or (energy_h is None):
                    if verbose:
                        print(f"[WARN] Missing data in last step of {path}")
                    pbar.update(1)
                    continue

                atoms = Atoms(symbols=symbols, positions=positions)
                energy_ev = energy_h * EV_PER_HARTREE
                calc = SinglePointCalculator(atoms, energy=energy_ev, forces=forces)
                atoms.set_calculator(calc)
                atoms.info['file'] = str(path)
                atoms.info['energy_dft'] = energy_ev
                if forces is not None:
                    atoms.arrays['forces_dft'] = forces

                atoms_list.append(atoms)

            except Exception as e:
                if verbose:
                    print(f"[ERROR] {path}: {e}")
            finally:
                pbar.update(1)

    return atoms_list


def parse_orca_dir_allsteps(filepaths, verbose, stepsize=1):
    """
    Find *.out files, parse **all** optimization steps and return a list of Atoms.
    Thinned by --stepsize applied per-file over steps.
    """
    atoms_list = []
    files = []
    for fp in filepaths:
        for root, _, fnames in os.walk(fp):
            for name in fnames:
                if name.lower().endswith('.out'):
                    files.append(Path(root) / name)

    with tqdm(total=len(files), desc='Processing ORCA (all steps)') as pbar:
        for path in files:
            try:
                content = Path(path).read_text(errors='ignore')
                coords_all = _orca_parse_all_coords(content)   # list of (symbols, pos)
                grads_all  = _orca_parse_all_gradients(content, to_ev_per_A=True)  # list of forces
                energies_all = _orca_parse_all_energies(content)  # list of Hartree

                n_steps = min(len(coords_all), len(grads_all)) if grads_all else len(coords_all)
                # Some ORCA outputs may have coords without gradients (e.g., SP); handle gracefully.
                for idx in range(0, n_steps, stepsize):
                    symbols, positions = coords_all[idx]
                    forces = grads_all[idx] if idx < len(grads_all) else None
                    energy_h = energies_all[idx] if idx < len(energies_all) else (energies_all[-1] if energies_all else None)

                    if (symbols is None) or (positions is None) or (energy_h is None):
                        if verbose:
                            print(f"[WARN] Missing data in {path} step {idx}")
                        continue

                    atoms = Atoms(symbols=symbols, positions=positions)
                    energy_ev = energy_h * EV_PER_HARTREE
                    calc = SinglePointCalculator(atoms, energy=energy_ev, forces=forces)
                    atoms.set_calculator(calc)
                    atoms.info['file'] = f"{path}::step{idx}"
                    atoms.info['energy_dft'] = energy_ev
                    if forces is not None:
                        atoms.arrays['forces_dft'] = forces

                    atoms_list.append(atoms)

            except Exception as e:
                if verbose:
                    print(f"[ERROR] {path}: {e}")
            finally:
                pbar.update(1)

    return atoms_list


# ---------------------------
# Post-processing utilities
# ---------------------------
def filter_atoms_list(atoms_list):
    """Clean list: allow missing stress (ORCA), but require forces and negative total energy."""
    filtered_atoms_list = []
    for atoms in atoms_list:
        # allow missing stress
        try:
            _ = atoms.get_stress()
        except PropertyNotImplementedError:
            pass

        has_forces = False
        try:
            f = atoms.get_forces()
            has_forces = f is not None and np.size(f) > 0
        except Exception:
            has_forces = False

        try:
            energy_ok = atoms.get_total_energy() < 0
        except Exception:
            energy_ok = False

        if has_forces and energy_ok:
            filtered_atoms_list.append(atoms)
    return filtered_atoms_list


def create_property_lists(atoms_list):
    """Return properties with tolerant stress handling."""
    total_energy = [atom.get_total_energy() for atom in atoms_list]
    forces = [np.array(atom.get_forces()) for atom in atoms_list]

    stresses = []
    for atom in atoms_list:
        try:
            stresses.append(np.array(atom.get_stress(voigt=False)))
        except PropertyNotImplementedError:
            stresses.append(None)

    mag_mom = None
    try:
        mag_mom = [np.array(atom.get_magnetic_moments()) for atom in atoms_list]
    except Exception as e:
        print(e)

    num_atoms_list = [atom.get_global_number_of_atoms() for atom in atoms_list]
    energies_per_atom = [energy / num_atoms for energy, num_atoms in zip(total_energy, num_atoms_list)]

    # annotate atoms
    for atom, energy in zip(atoms_list, total_energy):
        atom.info['energy_dft'] = energy
        atom.arrays['forces_dft'] = atom.get_forces()

    # NOTE: your original return order in the function conflicted with how you unpacked it in main().
    # To keep your current main() unpacking, we return in the order it expects:
    # (atoms_list, total_energy, energies_per_atom, forces, stresses, mag_mom)
    return (atoms_list, total_energy, energies_per_atom, forces, stresses, mag_mom)


def remove_outliers_quartile(atoms_list, energy_per_atom, threshold=None):
    q1 = np.percentile(energy_per_atom, 25)
    q3 = np.percentile(energy_per_atom, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    filtered_atoms_iqr = [atom for atom, energy in zip(atoms_list, energy_per_atom)
                          if lower_bound <= energy <= upper_bound]

    if threshold is not None:
        filtered_atoms_threshold = [atom for atom, energy in zip(filtered_atoms_iqr, energy_per_atom)
                                    if energy <= threshold]
        print('Outliers removed from data after threshold filtering:',
              len(filtered_atoms_iqr) - len(filtered_atoms_threshold))
        return filtered_atoms_threshold
    else:
        print('Outliers removed from data after quartile filtering:',
              len(atoms_list) - len(filtered_atoms_iqr))
        return filtered_atoms_iqr


def calculate_stats(energy_list):
    std_energy = np.std(energy_list)
    mean_energy = np.mean(energy_list)
    print(f"Your mean is {mean_energy:.3g} with std. {std_energy:.3g}")
    return (std_energy, mean_energy)


def graph_distribution(energies_per_atom):
    plt.hist(energies_per_atom, bins=5, label='Energy per Atom')
    q1 = np.percentile(energies_per_atom, 25)
    q2 = np.percentile(energies_per_atom, 50)
    q3 = np.percentile(energies_per_atom, 75)
    plt.axvline(q1, color='green', linestyle='dashed', linewidth=2, label='Q1')
    plt.axvline(q2, color='blue', linestyle='dashed', linewidth=2, label='Q2 (Median)')
    plt.axvline(q3, color='purple', linestyle='dashed', linewidth=2, label='Q3')
    plt.xlabel('Energy per Atom (eV)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Energy per Atom')
    plt.legend()
    plt.show()


def graph_filtered_distribution(energies_per_atom, energies_per_atom2):
    mean_energy = np.mean(energies_per_atom)
    std_energy = np.std(energies_per_atom)
    q1 = np.percentile(energies_per_atom, 25)
    q2 = np.percentile(energies_per_atom, 50)
    q3 = np.percentile(energies_per_atom, 75)

    mean_energy2 = np.mean(energies_per_atom2)
    std_energy2 = np.std(energies_per_atom2)
    q1_2 = np.percentile(energies_per_atom2, 25)
    q2_2 = np.percentile(energies_per_atom2, 50)
    q3_2 = np.percentile(energies_per_atom2, 75)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].hist(energies_per_atom, bins=5, label='Energy per Atom')
    axes[0].axvline(q1, color='green', linestyle='dashed', linewidth=2, label='Q1')
    axes[0].axvline(q2, color='blue', linestyle='dashed', linewidth=2, label='Q2 (Median)')
    axes[0].axvline(q3, color='purple', linestyle='dashed', linewidth=2, label='Q3')
    axes[0].set_xlabel('Energy per Atom (eV)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Energy per Atom')
    axes[0].legend()
    textstr = f'Mean: {mean_energy:.2f}\nStd Dev: {std_energy:.2f}'
    axes[0].text(0.7, 0.75, textstr, transform=axes[0].transAxes, fontsize=10, verticalalignment='top')

    axes[1].hist(energies_per_atom2, bins=5, label='Energy per Atom 2')
    axes[1].axvline(q1_2, color='green', linestyle='dashed', linewidth=2, label='Q1')
    axes[1].axvline(q2_2, color='blue', linestyle='dashed', linewidth=2, label='Q2 (Median)')
    axes[1].axvline(q3_2, color='purple', linestyle='dashed', linewidth=2, label='Q3')
    axes[1].set_xlabel('Energy per Atom (eV)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Energy per Atom after Filtering')
    textstr2 = f'Mean: {mean_energy2:.2f}\nStd Dev: {std_energy2:.2f}'
    axes[1].text(0.7, 0.75, textstr2, transform=axes[1].transAxes, fontsize=10, verticalalignment='top')
    axes[1].legend()
    plt.show()


def atoms_to_struct(atoms_list):
    structures = []
    for atom in atoms_list:
        struct = AseAtomsAdaptor().get_structure(atom)
        structures.append(struct)
    return structures


def properties_to_dict(structures, total_energy, energies_per_atom, forces, stresses, mag_mom=None):
    data_dict = {
        'structure': structures,
        'total_energy': total_energy,
        'energies_per_atom': energies_per_atom,
        'forces': forces,
        'stresses': stresses
    }
    if mag_mom is not None:
        data_dict['mag_mom'] = mag_mom
    return data_dict


def write_pickle(dataset_dict, output):
    with open(output, "wb") as f:
        pickle.dump(dataset_dict, f)


def prepare_data(output, atoms_list, total_energy, energies_per_atom, forces, stresses, mag_mom=None):
    structures = atoms_to_struct(atoms_list)
    if mag_mom is not None:
        dataset_dict = properties_to_dict(structures, total_energy, energies_per_atom, forces, stresses, mag_mom)
    else:
        dataset_dict = properties_to_dict(structures, total_energy, energies_per_atom, forces, stresses)
    write_pickle(dataset_dict, output)
    print(f"Total number of structures parsed {len(energies_per_atom)}")
    print('DONE!')


def prepare_mace(output, atoms_list):
    """
    Prepares data for MACE training by splitting dataset into training/test sets
    and writing to XYZ files. Skips inconsistent frames.
    """
    corrected_atoms_list = []
    for i, atoms in enumerate(atoms_list):
        expected_atoms = atoms.get_global_number_of_atoms()
        actual_atoms = len(atoms)
        if expected_atoms != actual_atoms:
            print(f"Warning: Frame {i} has {actual_atoms} atoms but expected {expected_atoms}. Skipping...")
            continue
        corrected_atoms_list.append(atoms)

    if len(corrected_atoms_list) < 2:
        raise ValueError("Not enough valid structures to create training and test sets.")

    train_data, test_data = train_test_split(corrected_atoms_list, test_size=0.1, random_state=42)

    print(f"Final dataset: {len(corrected_atoms_list)} structures")
    print(f"Training data: {len(train_data)}, Test data: {len(test_data)}")

    base = str(output).rsplit('.', 1)[0] if '.' in str(output) else str(output)
    train_file = f"{base}_train.xyz"
    test_file  = f"{base}_test.xyz"

    write(train_file, train_data)
    write(test_file, test_data)
    print(f"XYZ files written: {train_file}, {test_file}")


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    output = Path(args.output)
    filepaths = [Path(fp) for fp in args.input_filepaths]
    filter_flag = args.filter
    graph_flag = args.graph
    verbose_flag = args.verbose
    mace_flag = args.mace
    stepsize = max(1, int(args.stepsize))

    # Parse directories
    if args.orca_all_steps:
        atoms_list = parse_orca_dir_allsteps(filepaths, verbose_flag, stepsize=stepsize)
    elif args.orca:
        atoms_list = parse_orca_dir_laststep(filepaths, verbose_flag, stepsize=stepsize)
    else:
        atoms_list = parse_vasp_dir(filepaths, verbose_flag, stepsize=stepsize)

    # Filter & properties
    atoms_list = filter_atoms_list(atoms_list)

    # NOTE: the return order here matches the unpacking below.
    atoms_list, total_energy, energies_per_atom, forces, stresses, mag_mom = create_property_lists(atoms_list)
    calculate_stats(energies_per_atom)

    # Keep copy of original (pre-filter) distribution for graphing
    energies_per_atom2 = energies_per_atom.copy()

    if filter_flag:
        atoms_list = remove_outliers_quartile(atoms_list, energies_per_atom)
        atoms_list, total_energy, energies_per_atom, forces, stresses, mag_mom = create_property_lists(atoms_list)
        calculate_stats(energies_per_atom)

    if graph_flag:
        if filter_flag:
            graph_filtered_distribution(energies_per_atom2, energies_per_atom)
        else:
            graph_distribution(energies_per_atom)

    if mace_flag:
        prepare_mace(output, atoms_list)
    else:
        prepare_data(output, atoms_list, total_energy, energies_per_atom, forces, stresses, mag_mom)


if __name__ == '__main__':
    main()

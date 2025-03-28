from ase import Atoms
from ase.io import read,write
import os
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from chgnet.utils import parse_vasp_dir
import warnings
from pymatgen.io.ase import AseAtomsAdaptor
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from ase.calculators.calculator import PropertyNotImplementedError


def parse_args():
    parser = argparse.ArgumentParser(description="Parse VASP directories and process atomic data")
    parser.add_argument('--input_filepaths', nargs='+', required=True, help="Paths to input directories")
    parser.add_argument('--output', required=True, help="Output directory")
    parser.add_argument('--filter', action='store_true', help="Apply outlier filtering")
    parser.add_argument('--graph', action='store_true', help="Generate graphs")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose output")
    parser.add_argument('--mace', action='store_true', help="Prepare MACE-compatible data")
    parser.add_argument('--stepsize', type=int, default=1, help="Sampling step size")

    return parser.parse_args()

def parse_vasp_dir(filepaths, verbose, stepsize=1):
    """Parse multiple VASP directories and extract relevant atomic structures."""
    warnings.filterwarnings('ignore')
    atoms_list = []
    len_list = []

    # Determine total iterations across all directories
    total_iterations = sum(len(os.listdir(fp)) for fp in filepaths)

    with tqdm(total=total_iterations, desc='Processing') as pbar:
        for filepath in filepaths:
            for i in os.listdir(filepath):
                dir = Path(i)
                outcar_path = filepath / i / 'OUTCAR'
                pwscf_path = filepath / i / 'pwscf.out'

                if outcar_path.exists():
                    file_path = outcar_path
                elif pwscf_path.exists():
                    file_path = pwscf_path
                else:
                    if verbose:
                        print(f"No valid file ('OUTCAR' or 'pwscf.out') found in {filepath / i}")
                    pbar.update(1)
                    continue

                try:
                    # Read the entire file (OUTCAR or pwscf.out)
                    single_file_atom = read(file_path, index=':')
                    last_energy = read(file_path)
                    last_energy = last_energy.get_total_energy()

                    # Convert single_file_atom to a list
                    all_steps = list(single_file_atom)
                    len_list.append(len(all_steps))

                    # Sample atoms from all_steps based on stepsize
                    sampled_atoms = all_steps[::stepsize]

                    # Process sampled atoms
                    for a in sampled_atoms:
                        if a.get_total_energy() < 0:
                            a.info['file'] = filepath.joinpath(i)
                            atoms_list.append(a)
                except Exception as e:
                    if verbose:
                        print(f"Error reading file: {file_path}")
                        print(f"Error details: {e}")
                    continue
                finally:
                    pbar.update(1)

    return atoms_list

def filter_atoms_list(atoms_list):
    """This is a function to clean the list if there are any missing values"""
    filtered_atoms_list = []
    
    for atoms in atoms_list:
        try:
            # Check if stress is available
            stress = atoms.get_stress()
        except PropertyNotImplementedError:
            # If stress is not available, skip this atoms object
            continue
        
        # Check if forces are available and not empty, and if the total energy is less than 0
        if atoms.get_forces().any() and atoms.get_total_energy() < 0:
            filtered_atoms_list.append(atoms)
            
    return filtered_atoms_list

def create_property_lists(atoms_list):
        #Energy lists for every optimization step 
        total_energy = []
        total_energy = [atom.get_total_energy() for atom in atoms_list]
        forces = [np.array(atom.get_forces()) for atom in atoms_list]
        stresses = [np.array(atom.get_stress(voigt=False)) for atom in atoms_list]
        mag_mom = None  # Initialize mag_mom to None

        try: 
            mag_mom = [np.array(atom.get_magnetic_moments()) for atom in atoms_list]
        except Exception as e:
                print(e)
        num_atoms_list = [atom.get_global_number_of_atoms() for atom in atoms_list]
        energies_per_atom = [energy / num_atoms for energy, num_atoms in zip(total_energy, num_atoms_list)]
        std_energy_per_atom = np.std(energies_per_atom)

      # Add total energy to atom.info as energy_dft and rename forces as forces_dft
        for atom, energy in zip(atoms_list, total_energy):
            atom.info['energy_dft'] = energy
            atom.arrays['forces_dft'] = atom.get_forces() 
        
        return(atoms_list,energies_per_atom,total_energy,forces,stresses,mag_mom)

def remove_outliers_quartile(atoms_list, energy_per_atom, threshold=None):
    q1 = np.percentile(energy_per_atom, 25)
    q3 = np.percentile(energy_per_atom, 75)

    # Calculate the interquartile range (IQR)
    iqr = q3 - q1

    # Define the range for filtering outliers based on IQR
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Filter atoms based on the IQR range
    filtered_atoms_iqr = [atom for atom, energy in zip(atoms_list, energy_per_atom)
                          if lower_bound <= energy <= upper_bound]

    # If threshold is provided, filter based on threshold
    if threshold is not None:
        filtered_atoms_threshold = [atom for atom, energy in zip(filtered_atoms_iqr, energy_per_atom)
                                    if energy <= threshold]
        outliers_removed_threshold = len(filtered_atoms_iqr) - len(filtered_atoms_threshold)
        print('Outliers removed from data after threshold filtering:', outliers_removed_threshold)
        return filtered_atoms_threshold
    else:
        print('Outliers removed from data after quartile filtering:', len(atoms_list) - len(filtered_atoms_iqr))
        return filtered_atoms_iqr

def calculate_stats(energy_list):
      std_energy = np.std(energy_list)
      mean_energy = np.mean(energy_list)
      print(f"Your mean is {mean_energy:.3g} with std. {std_energy:.3g}")
      return(std_energy,mean_energy)

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

def graph_filtered_distribution(energies_per_atom,energies_per_atom2):
    # Example data
   
    # Calculate statistics for the first dataset
    mean_energy = np.mean(energies_per_atom)
    std_energy = np.std(energies_per_atom)
    q1 = np.percentile(energies_per_atom, 25)
    q2 = np.percentile(energies_per_atom, 50)
    q3 = np.percentile(energies_per_atom, 75)
    
# Calculate statistics for the second dataset
    mean_energy2 = np.mean(energies_per_atom2)
    std_energy2 = np.std(energies_per_atom2)
    q1_2 = np.percentile(energies_per_atom2, 25)
    q2_2 = np.percentile(energies_per_atom2, 50)
    q3_2 = np.percentile(energies_per_atom2, 75)

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot histogram for the first dataset
    counts, bins, _ = axes[0].hist(energies_per_atom, bins=5, label='Energy per Atom')
    axes[0].axvline(q1, color='green', linestyle='dashed', linewidth=2, label='Q1')
    axes[0].axvline(q2, color='blue', linestyle='dashed', linewidth=2, label='Q2 (Median)')
    axes[0].axvline(q3, color='purple', linestyle='dashed', linewidth=2, label='Q3')
    axes[0].set_xlabel('Energy per Atom (eV)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Energy per Atom')
    axes[0].legend()

    # Add text for mean and std dev
    textstr = f'Mean: {mean_energy:.2f}\nStd Dev: {std_energy:.2f}'
    axes[0].text(0.7, 0.75, textstr, transform=axes[0].transAxes, fontsize=10, verticalalignment='top')

    # Plot histogram for the second dataset
    counts2, bins2, _ = axes[1].hist(energies_per_atom2, bins=5, color='orange', label='Energy per Atom 2')
    axes[1].axvline(q1_2, color='green', linestyle='dashed', linewidth=2, label='Q1')
    axes[1].axvline(q2_2, color='blue', linestyle='dashed', linewidth=2, label='Q2 (Median)')
    axes[1].axvline(q3_2, color='purple', linestyle='dashed', linewidth=2, label='Q3')
    axes[1].set_xlabel('Energy per Atom (eV)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Energy per Atom after Filtering')
    textstr2 = f'Mean: {mean_energy2:.2f}\nStd Dev: {std_energy2:.2f}'
    axes[1].text(0.7, 0.75, textstr2, transform=axes[1].transAxes, fontsize=10, verticalalignment='top')
    axes[1].legend()
    # Display the plot
    plt.show()

def atoms_to_struct(atoms_list):
    structures = []
    for atom in atoms_list:
        struct = AseAtomsAdaptor().get_structure(atom)
        structures.append(struct) 
    return(structures)

def properties_to_dict(structures, total_energy,energies_per_atom, forces, stresses,mag_mom=None):
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


def write_pickle(dataset_dict,output):
     with open(output, "wb") as f:
        pickle.dump(dataset_dict, f)

def prepare_data(output, atoms_list, energies_per_atom, total_energy,forces, stresses, mag_mom=None):
    structures = atoms_to_struct(atoms_list)
    
    if mag_mom is not None:
        dataset_dict = properties_to_dict(structures, total_energy, energies_per_atom, forces, stresses,mag_mom)
    else:
        dataset_dict = properties_to_dict(structures, total_energy,energies_per_atom, forces,stresses)
    
    write_pickle(dataset_dict, output)
    print(f"Total number of structures parsed {len(energies_per_atom)}")
    print('DONE!')

def prepare_mace(output, atoms_list):
    """
    Prepares data for MACE training by splitting the dataset into training and test sets,
    ensuring the atom counts are correct, and writing the data to XYZ files.
    """
    # Check atom counts before proceeding
    corrected_atoms_list = []
    for i, atoms in enumerate(atoms_list):
        expected_atoms = atoms.get_global_number_of_atoms()
        actual_atoms = len(atoms)
        
        if expected_atoms != actual_atoms:
            print(f"Warning: Frame {i} has {actual_atoms} atoms but expected {expected_atoms}. Skipping...")
            continue  # Skip inconsistent frames
        
        corrected_atoms_list.append(atoms)

    # Ensure we have enough valid frames to split
    if len(corrected_atoms_list) < 2:
        raise ValueError("Not enough valid structures to create training and test sets.")

    # Split data into training and test sets
    train_data, test_data = train_test_split(corrected_atoms_list, test_size=0.1, random_state=42)

    print(f"Final dataset: {len(corrected_atoms_list)} structures")
    print(f"Training data: {len(train_data)}, Test data: {len(test_data)}")

    # Determine output file names
    base = str(output).rsplit('.', 1)[0] if '.' in str(output) else str(output)
    train_file = f"{base}_train.xyz"
    test_file = f"{base}_test.xyz"

    # Write to XYZ files
    write(train_file, train_data)
    write(test_file, test_data)

    print(f"XYZ files written: {train_file}, {test_file}")

    
def main(): 
    args = parse_args()
    output = Path(args.output)
    filepaths = [Path(fp) for fp in args.input_filepaths]  # Accept multiple paths
    filter_flag = args.filter
    graph_flag = args.graph
    verbose_flag = args.verbose
    mace_flag = args.mace
    stepsize = args.stepsize  

    # Parse all provided directories
    atoms_list = parse_vasp_dir(filepaths, verbose_flag, stepsize=stepsize)
    atoms_list = filter_atoms_list(atoms_list)

    # Initial property extraction
    atoms_list, total_energy, energies_per_atom, forces, stresses, mag_mom = create_property_lists(atoms_list)
    calculate_stats(energies_per_atom)

    # Save original energies before filtering
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

      

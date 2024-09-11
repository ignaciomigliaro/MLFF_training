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
    parser = argparse.ArgumentParser(
        description="This is a code to parse for Chgnet and have fast parsing"
    )
    parser.add_argument(
        "input_filepath",
        help="Directory that contains all of your VASP files"
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="If specified, generate a graph of the Variability of the data"
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="If specified, remove outliers in the data"
    )
    parser.add_argument(
        "--relaxed",
        action="store_true",
        help="If you want to only parse relaxed energy and have structures predict relaxed energies"
    )
    parser.add_argument(
        "output",
        help="If you want to only parse relaxed energy and have structures predict relaxed energies"
    )
    parser.add_argument(
         '--mace',
         action='store_true',
         help="If you want to parse for MACE and create an XYZ file instead of a pickle file"
    )
    parser.add_argument(
         '--verbose',
         action='store_true',
         help="If you want to see any error occurring in the parsing process"

    )
    parser.add_argument(
        '--stepsize',
        type=int,
        default=1,  # Default value is 1 so it behaves like before if not provided
        help="Step size for selecting atoms every nth step (default: 1)"
    )
    return parser.parse_args()

def parse_vasp_dir(filepath, verbose, stepsize=1):
    """This is a function to replace the Chgnet Utils, this is more robust and gives correct energy values"""
    warnings.filterwarnings('ignore')
    atoms_list = []
    len_list = []
    filename = Path('OUTCAR')
    total_iterations = len(os.listdir(filepath))

    with tqdm(total=total_iterations, desc='Processing') as pbar:
        for i in os.listdir(filepath):
            dir = Path(i)
            OUTCAR = filepath / i / filename
            try:
                # Read the entire OUTCAR file
                single_file_atom = read(OUTCAR, format='vasp-out', index=':')
                last_energy = read(OUTCAR, format='vasp-out')
                last_energy = last_energy.get_total_energy()
                all_steps = list(single_file_atom)
                len_list.append(len(all_steps))

                # Ensure stepsize is within the bounds of the list
                if stepsize > len(all_steps):
                    stepsize = len(all_steps)

                # Process only every 'stepsize'-th atom object
                for idx in range(0, len(all_steps), stepsize):
                    a = all_steps[idx]
                    if a.get_total_energy() < 0:
                        a.info['file'] = filepath.joinpath(i)
                        atoms_list.append(a)
                        e = a.get_total_energy()
                        a.info['relaxed_energy'] = last_energy
            except Exception as e:
                if verbose:
                    print(f"Error reading file: {OUTCAR}")
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

        #Energy lists for relaxed energies
        relaxed_total_energy = []
        relaxed_total_energy = [atom.info['relaxed_energy'] for atom in atoms_list]
        num_atoms_list = [atom.get_global_number_of_atoms() for atom in atoms_list]
        relaxed_energies_per_atom = [energy / num_atoms for energy, num_atoms in zip(relaxed_total_energy, num_atoms_list)]
        for atom, relaxed_energy_per_atom in zip(atoms_list, relaxed_energies_per_atom):
            atom.info['relaxed_energy_per_atom'] = relaxed_energy_per_atom

        return(atoms_list,energies_per_atom,forces,stresses,relaxed_energies_per_atom,mag_mom)

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

def properties_to_dict(structures, energies_per_atom, forces, stresses, relaxed_energies_per_atom,mag_mom=None):
    data_dict = {
        'structure': structures,
        'energies_per_atom': energies_per_atom,
        'forces': forces,
        'relaxed_energies':relaxed_energies_per_atom,
        'stresses': stresses
    }
    
    if mag_mom is not None:
        data_dict['mag_mom'] = mag_mom
    
    return data_dict


def write_pickle(dataset_dict,output):
     with open(output, "wb") as f:
        pickle.dump(dataset_dict, f)

def prepare_data(output, atoms_list, energies_per_atom, forces, stresses,relaxed_energies_per_atom, mag_mom=None):
    structures = atoms_to_struct(atoms_list)
    
    if mag_mom is not None:
        dataset_dict = properties_to_dict(structures, energies_per_atom, forces, stresses,relaxed_energies_per_atom,mag_mom)
    else:
        dataset_dict = properties_to_dict(structures, energies_per_atom, forces,stresses,relaxed_energies_per_atom)
    
    write_pickle(dataset_dict, output)
    print(f"Total number of structures parsed {len(energies_per_atom)}")
    print('DONE!')

def prepare_mace(output,atoms_list):
     train_data, test_data = train_test_split(atoms_list, test_size=0.1, random_state=42)
     print(f"Your number of data is {len(atoms_list)} training data is {len(train_data)} and test data {len(test_data)}")
     if '.' in str(output):
          base,old_extension = str(output).rsplit('.',1)
          train_file = f"{base}_train.xyz"
          test_file = f"{base}_test.xyz"
     else: 
          train_file = f"{base}_train.xyz"
          test_file = f"{base}_test.xyz"
     write(train_file,train_data)
     write(test_file,test_data)

     

def main(): 
    args = parse_args()
    output = Path(args.output)
    filepath = Path(args.input_filepath)
    filter_flag = args.filter
    graph_flag = args.graph
    relax_flag = args.relaxed
    verbose_flag = args.verbose
    mace_flag = args.mace
    stepsize = args.stepsize  # Capture stepsize from parsed arguments


    atoms_list = parse_vasp_dir(filepath, verbose_flag, stepsize=stepsize)
    atoms_list = filter_atoms_list(atoms_list)
    
    # Initial property extraction
    atoms_list, energies_per_atom, forces, stresses, relaxed_energies_per_atom, mag_mom = create_property_lists(atoms_list)
    calculate_stats(relaxed_energies_per_atom if relax_flag else energies_per_atom)
    
    energy = relaxed_energies_per_atom if relax_flag else energies_per_atom
    
    if filter_flag:
        energies_per_atom2 = energy
        atoms_list = remove_outliers_quartile(atoms_list, energy)
        atoms_list, energies_per_atom, forces, stresses, relaxed_energies_per_atom, mag_mom = create_property_lists(atoms_list)
        calculate_stats(relaxed_energies_per_atom if relax_flag else energies_per_atom)
        energy = relaxed_energies_per_atom if relax_flag else energies_per_atom

    if graph_flag:
        if filter_flag:
            if relax_flag:
                graph_filtered_distribution(energies_per_atom2, relaxed_energies_per_atom)
            else:
                graph_filtered_distribution(energies_per_atom2, energies_per_atom)
        else:
            if relax_flag:
                graph_distribution(relaxed_energies_per_atom)
            else:
                graph_distribution(energies_per_atom)

    if mace_flag: 
        prepare_mace(output, atoms_list)
    else:
        prepare_data(output, atoms_list, energy, forces, stresses, mag_mom)

if __name__ == '__main__':
    main()

      

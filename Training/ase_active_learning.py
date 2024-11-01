from ase.io import read,write
import os
import argparse
from chgnet.model.dynamics import CHGNetCalculator
from chgnet.model import CHGNet
import contextlib
import numpy as np 
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import torch

"""
This code is for running an agnostic active learning workflow that works on any MLFF that has an ASE calculator and configuration space generated by molecular dynamics. 
The idea is to have a general workflow that can be used to benchmark any MLFF.
"""

# A dictionary to map calculator names to actual ASE calculator classes
# TO-DO make active learning work on all types of MLFF
CALCULATOR_MAP = {
    'ALIGNN': 'alignn.ase.ALIGNN',  # String representation of the import path
    'CHGnet': 'from chgnet.model.dynamics import CHGNetCalculator',
    'DEEPMD-kit': 'deepmd.ase.DeepMD',
    'MG3NET': 'mg3net.ase.MG3Net',
    'MACE': 'mace.ase.MACE',
}
    
def parse_arguments():
        parser = argparse.ArgumentParser(description="Active learning with MLFF for crystal structures.")
        parser.add_argument("--filepath", type=str, required=True, help="Path to the configuration file or directory.")
        parser.add_argument("--stepsize", type=int, default=1, help="Step size for loading configurations.")
        parser.add_argument("--model_dir", type=str, required=True, help="Directory containing trained models.")
        parser.add_argument("--calculator", type=str, required=True, help="Calculator to use (e.g., chgnet, some_other_calculator).")
        parser.add_argument("--device", type=str, default="cpu", help="Device to use for computation (e.g., cpu or cuda).")
        parser.add_argument("--dft_software", type=str, choices=["qe"], required=False,help="DFT software to use. Currently only 'quantum_espresso' is supported.")
        parser.add_argument("--threshold", type=float, default=None, help="User-defined threshold for filtering structures.")
        parser.add_argument("--plot_std_dev", action="store_true", help="Flag to plot the distribution of standard deviations.")
        parser.add_argument("--output_dir", type=str, default="qe_outputs", help="Directory to save Quantum Espresso files.")
        parser.add_argument('--use_cache', type=str, default=None, help='Path to cache file for storing/loading energy and std_dev data')
        return parser.parse_args()

def get_configuration_space(path, stepsize):
    """
    Reads configurations from a file or all files in a directory using ASE's read function.

    Parameters:
    - path (str): The path to a file or directory. If a directory, all files inside will be read.
    - stepsize (int): The step size for subsampling configurations.

    Returns:
    - configurations (list): A list of ASE Atoms objects representing different configurations.
    """
    configurations = []

    if os.path.isfile(path):
        # If it's a single file, read all configurations from that file
        all_atoms = read(path, index=':')
        configurations = all_atoms[::stepsize]  # Subsample using the stepsize
    elif os.path.isdir(path):
        # If it's a directory, iterate through all files
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                # Read configurations from each file and subsample
                all_atoms = read(file_path, index=':')
                configurations.extend(all_atoms[::stepsize])  # Subsample using the stepsize
    else:
        raise ValueError(f"The provided path '{path}' is neither a file nor a directory.")
    
    return configurations

def load_models(model_dir, device='cpu', extension='.pth.tar'):
    """
    Load CHGNet models from a specified directory, suppress output, and handle loading errors.

    Parameters:
    - model_dir (str): Directory containing model files.
    - device (str): Device to use for loading models (e.g., 'cpu' or 'cuda').
    - extension (str): File extension to filter model files (default is '.pth.tar').

    Returns:
    - models (list): List of loaded CHGNet models.
    """
    models = []
    for filename in os.listdir(model_dir):
        if filename.endswith(extension):
            model_path = os.path.join(model_dir, filename)
            try:
                # Suppress output during model loading
                with open(os.devnull, 'w') as fnull:
                    with contextlib.redirect_stdout(fnull):
                        loaded_model = CHGNet.from_file(
                            model_path,
                            use_device=device,
                            check_cuda_mem=False,
                            verbose=False
                        )
                models.append(loaded_model)
            except Exception as e:
                print(f"Failed to load model from {model_path}: {e}")

    print(f"Successfully loaded {len(models)} models.")
    return models

def calculate_properties(configurations, models, device='cpu', cache_file=None):
    """
    Create a list of configurations for each model, assign calculators, and optionally cache the results.

    Parameters:
    - configurations (list): Original list of ASE Atoms objects (the base configuration).
    - models (list): List of CHGNet models to use for creating configurations.
    - device (str): Device to use for calculation (e.g., 'cpu' or 'cuda').
    - cache_file (str, optional): Path to a file for saving/loading the configurations. If None, no caching is done.

    Returns:
    - all_configurations (list): List of lists where each inner list corresponds to configurations for a single model.
    """
    all_configurations = []

    for model in tqdm(models, desc="Models"):
        calculator = CHGNetCalculator(model=model, use_device=device)

        # Create a deep copy of configurations and assign calculator for this model
        config_copy = [atoms.copy() for atoms in configurations]
        for atoms in tqdm(config_copy, desc="Configurations", leave=False):
            atoms.calc = calculator  # Set the model-specific calculator

        all_configurations.append(config_copy)

    # Cache the configurations if a cache_file is specified
    if cache_file:
        try:
            data_to_save = {'all_configurations': all_configurations}
            torch.save(data_to_save, cache_file)
            print(f"Configurations saved to {cache_file}.")

            # Reload in CPU mode and re-save
            cpu_data = torch.load(cache_file, map_location='cpu')
            torch.save(cpu_data, cache_file)
            print(f"Configurations re-saved to {cache_file} in CPU-compatible format.")

        except Exception as e:
            print(f"Error processing cache file: {e}")

    return all_configurations

def calculate_std_dev(all_configurations):
    """
    Calculate the standard deviation of energies for each atom across different models.

    Parameters:
    - all_configurations (list): List of configurations, where each configuration is a list of ASE Atoms objects
                                 with calculators already set for different models.

    Returns:
    - std_dev (list): A list containing the standard deviation of energies for each atom across the models.
    """
    num_atoms = len(all_configurations[0])  # Assume all configurations have the same number of atoms
    energies = [[] for _ in range(num_atoms)]

    # Iterate through each configuration (model) and get energies for each atom
    for config in all_configurations:
        for i, atom in enumerate(config):
            energy = atom.get_total_energy()
            energies[i].append(energy)

    # Calculate standard deviation across configurations for each atom
    energies_array = np.array(energies)
    std_dev = np.std(energies_array, axis=1)

    return std_dev.tolist()

def filter_high_deviation_structures(atoms_lists, std_dev, user_threshold=None, percentile=90):
    """
    Filters structures based on the normalized standard deviation.
    Includes structures with normalized deviation equal to or above the specified threshold.

    Parameters:
    - atoms_lists (list of list of ASE Atoms): List containing multiple atoms lists for each model.
    - energies (list of list of floats): List containing energies for each model.
    - std_dev (list of floats): Standard deviation values.
    - user_threshold (float, optional): User-defined threshold for filtering. If None, percentile-based threshold is used.
    - percentile (int): Percentile threshold for filtering if no user threshold is provided.

    Returns:
    - filtered_atoms_list (list of ASE Atoms): List of filtered structures.
    """
    # Compute the normalized standard deviation
    std_dev_normalized = [std / len(atoms_lists[0]) for std in std_dev]
    #print(std_dev_normalized)
    # Determine the threshold
    if user_threshold is not None:
        threshold = float(user_threshold)
        print(f"User-defined threshold for filtering: {threshold}")
    else:
        threshold = np.percentile(std_dev_normalized, percentile)
        print(f"Threshold for filtering (95th percentile): {threshold}")

    # Filter structures based on the chosen threshold
    filtered_atoms_list = []
    for i, norm_dev in enumerate(std_dev_normalized):
        if norm_dev >= threshold:  # Include structures with deviation >= threshold
            filtered_atoms_list.append(atoms_lists[0][i])
    return filtered_atoms_list

def plot_std_dev_distribution(std_devs):
    """
    Plots the distribution of standard deviations using a histogram.

    Parameters:
    - std_devs (list): List of standard deviation values to plot.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(std_devs, bins=20, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Standard Deviations')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Frequency')
    plt.axvline(x=np.percentile(std_devs, 98), color='r', linestyle='--', label='98th Percentile')
    plt.legend()
    plt.grid(True)
    plt.show()

def write_qe_file(output_directory, crystal_structure):
    # Define QE input parameters
    input_data = {
        "calculation": "scf",
        "prefix": "qe_input",
        "pseudo_dir": "~/QE/pseudo",
        "outdir": "./out/",
        "verbosity": "high",
        "etot_conv_thr": 1.0e-03,
        "tstress": True,
        "tprnfor": True,
        "degauss": 1.4699723600e-02,
        "ecutrho": 600,
        "ecutwfc": 90,
        "vdw_corr": "mbd",
        "occupations": "smearing",
        "smearing": 'cold',
        "electron_maxstep": 80,
        "mixing_beta": 4.0e-01,
    }
    
    # Define pseudopotentials
    pseudos = {
        "Cl": "Cl.upf",
        "O": "O.upf",
        "F": "F.upf",
        "I": "I.upf",
        "Br": "Br.upf",
        "La": "La.upf",
        "Li": "Li.upf",
        "Zr": "Zr.upf",
        "C" : "C.upf",
        "H" : "H.upf",
        "Nb": "Nb.upf",
    }

    # Create the output directory if it does not exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Write the QE input file using ASE's write function
    filename = os.path.join(output_directory, "qe_input.in")
    write(
        format='espresso-in',
        filename=filename,
        images=crystal_structure, 
        input_data=input_data,
        pseudopotentials=pseudos,
        kspacing=0.05
    )

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Get the configuration space based on the provided file or directory
    atoms_list = get_configuration_space(args.filepath, args.stepsize)

    # Load models from the specified directory
    models = load_models(args.model_dir)

    # Print the number of configurations loaded
    print(f"Number of configurations loaded: {len(atoms_list)}")

    # Calculate energies and standard deviation, using cache if specified
    atoms_list = calculate_properties(
        configurations=atoms_list,
        models=models,
        device=args.device,
        cache_file=args.use_cache  # Pass the user-defined cache file path or None if not provided
    )
    std_dev= calculate_std_dev(atoms_list)
    # Plot the distribution of standard deviations if the flag is set
    if args.plot_std_dev:
        plot_std_dev_distribution(std_dev)

    # Use user-defined threshold to filter high-deviation structures
    filtered_atoms_list = filter_high_deviation_structures(
        atoms_lists=atoms_list,
        std_dev=std_dev,
        user_threshold=args.threshold
    )
    
    print(f"Number of filtered structures: {len(filtered_atoms_list)}")

    # Write input files for the filtered structures based on DFT software choice
    for idx, atoms in enumerate(filtered_atoms_list):
        # Define the subdirectory for each filtered structure
        structure_output_dir = os.path.join(args.output_dir, f"structure_{idx}")
        os.makedirs(structure_output_dir, exist_ok=True)

        # Write the appropriate input file based on the chosen DFT software
        if args.dft_software.lower() == 'qe':
            write_qe_file(structure_output_dir, atoms)
        else:
            print(f"Unsupported DFT software: {args.dft_software}")

if __name__ == "__main__":
    main()

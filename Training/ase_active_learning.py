from ase.io import read
import os
import argparse
from chgnet.model.dynamics import CHGNetCalculator
from chgnet.model import CHGNet
import contextlib
import numpy as np 
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
    """
    Parses command-line arguments for the script.

    Returns:
    - args: The parsed arguments object with attributes for each argument.
    """
    parser = argparse.ArgumentParser(description="Read configurations and choose a calculator for ASE.")
    
    parser.add_argument(
        "filepath",
        type=str,
        help="Path to a file or directory containing the configuration files."
    )
    
    parser.add_argument(
        "--calculator",
        type=str,
        choices=CALCULATOR_MAP.keys(),
        required=True,
        default='CHGnet',
        help="Choose a calculator: ALIGNN, CHGnet, DEEPMD-kit, MG3NET, MACE."
    )

    parser.add_argument(
        "--stepsize",
        type=int,
        default=1,
        help="Step size for subsampling configurations (default: 1)."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the directory containing model files."
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help='Device used to calculate inference using MLFF'
    )
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

def load_models(model_dir, device="cpu"):
    """
    Load models from a specified directory.

    Parameters:
    - model_dir (str): Path to the directory containing the models.
    - device (str): Device to use for loading the model (e.g., "cpu" or "cuda").

    Returns:
    - list: A list of loaded model objects.
    """
    models = []
    for filename in os.listdir(model_dir):
        if filename.endswith('.pth.tar'):  # Adjust this based on your model file extension
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
    return models

def set_calculator(atoms_list, calculator_name, models, device):
    """
    Sets the calculator for a list of ASE Atoms objects based on user input.

    Parameters:
    - atoms_list (list): List of ASE Atoms objects.
    - calculator_name (str): The name of the calculator to use.
    - models (list): List of loaded models for the calculator.
    - device (str): Device used to calculate inference.

    Returns:
    - updated_atoms_lists (list): List of lists of ASE Atoms objects with calculators set.
    """
    updated_atoms_lists = []  # List to hold the updated atoms lists for each model

    if calculator_name == 'CHGnet':
        from chgnet.model.dynamics import CHGNetCalculator
        
        for model in models:
            # Create calculator instance for this model
            calculator = CHGNetCalculator(model=model, use_device=device)

            # Create a new atoms list for the current model
            atoms_copy = [atom.copy() for atom in atoms_list]
            for atom in atoms_copy:
                atom.calc = calculator  # Set the calculator for each Atoms object

            updated_atoms_lists.append(atoms_copy)

    return updated_atoms_lists

def calculate_energies_and_std(atoms_lists):
    """
    Calculate the energies for each atom in the configuration across different calculators
    and get the standard deviation of the energies divided by the number of atoms.

    Parameters:
    - atoms_lists (list): List of lists of ASE Atoms objects with calculators set.

    Returns:
    - energies (list): A list of lists containing energies for each atom across calculators.
    - std_dev (list): A list of standard deviations of the energies for each atom divided by N.
    """
    # Initialize a list to store energies for each configuration
    energies = []

    for atoms in atoms_lists:
        energy_for_atoms = []
        for atom in atoms:
            # Calculate the energy using the calculator
            energy = atom.get_potential_energy()  # This will call the calculator's energy method
            energy_for_atoms.append(energy)
        energies.append(energy_for_atoms)

    # Convert to numpy array for easier standard deviation calculation
    energies_array = np.array(energies)

    # Calculate standard deviation for each atom across all calculators
    std_dev = np.std(energies_array, axis=0)

    # Divide standard deviation by the number of atoms
    num_atoms = len(atoms_lists[0])  # Assuming all lists have the same number of atoms
    std_dev_normalized = std_dev / num_atoms

    return energies_array.tolist(), std_dev_normalized.tolist()

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Get the configuration space based on the provided file or directory
    atoms_list = get_configuration_space(args.filepath,args.stepsize)

    # Load models from the specified directory
    models = load_models(args.model_dir)

    # Set the calculator and get the atoms lists for each model
    atoms_lists = set_calculator(atoms_list, args.calculator, models,args.device)

    # Print the number of configurations loaded
    print(f"Number of configurations loaded: {len(atoms_list)}")
    energies, std_dev = calculate_energies_and_std(atoms_lists)
    print(len(std_dev))
    print(std_dev)

if __name__ == "__main__":
    main()

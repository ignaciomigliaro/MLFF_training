from ase.io import read
import os
import argparse

"""
This code is for running an agnostic active learning workflow that works on any MLFF that has an ASE calculator and configuration space generated by molecular dynamics. 
The idea is to have a general workflow that can be used to benchmark any MLFF.
"""

# A dictionary to map calculator names to actual ASE calculator classes
CALCULATOR_MAP = {
    'ALIGNN': 'alignn.ase.ALIGNN',  # String representation of the import path
    'CHGnet': 'chgnet.ase.CHGNet',
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
        "path",
        type=str,
        help="Path to a file or directory containing the configuration files."
    )
    
    parser.add_argument(
        "--calculator",
        type=str,
        choices=CALCULATOR_MAP.keys(),
        required=True,
        help="Choose a calculator: ALIGNN, CHGnet, DEEPMD-kit, MG3NET, MACE."
    )
    
    return parser.parse_args()

def get_configuration_space(path):
    """
    Reads configurations from a file or all files in a directory using ASE's read function.

    Parameters:
    - path (str): The path to a file or directory. If a directory, all files inside will be read.

    Returns:
    - configurations (list): A list of ASE Atoms objects representing different configurations.
    """
    configurations = []

    if os.path.isfile(path):
        # If it's a single file, read all configurations from that file
        configurations = read(path, index=':')
    elif os.path.isdir(path):
        # If it's a directory, iterate through all files
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                # Read configurations from each file and add to the list
                configurations.extend(read(file_path, index=':'))
    else:
        raise ValueError(f"The provided path '{path}' is neither a file nor a directory.")
    
    return configurations

def set_calculator(atoms_list, calculator_name):
    """
    Sets the calculator for a list of ASE Atoms objects based on user input.

    Parameters:
    - atoms_list (list): List of ASE Atoms objects.
    - calculator_name (str): The name of the calculator to use.

    Returns:
    - updated_atoms_list (list): List of ASE Atoms objects with the calculator set.
    """
    # Import the selected calculator dynamically
    module_name, class_name = CALCULATOR_MAP[calculator_name].rsplit('.', 1)
    module = __import__(module_name, fromlist=[class_name])
    CalculatorClass = getattr(module, class_name)
    
    # Apply the calculator to each Atoms object in the list
    for atoms in atoms_list:
        atoms.set_calculator(CalculatorClass())
    
    return atoms_list

def main():
    """
    Main function to handle the workflow.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Read configurations from the provided path
    configurations = get_configuration_space(args.path)
    
    # Set the chosen calculator for all configurations
    configurations = set_calculator(configurations, args.calculator)
    
    # Print the number of configurations read and the chosen calculator
    print(f"Total configurations read: {len(configurations)}")
    print(f"Calculator set to: {args.calculator}")

if __name__ == "__main__":
    main()
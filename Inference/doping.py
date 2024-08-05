import numpy as np
from ase.io import read, write
from itertools import combinations
import random
import os
import argparse

def introduce_dopant(structure, original_type, dopant, num_dopants, num_structures):
    """
    Introduce a specified number of dopants in place of the original atom type.

    Parameters:
    structure (ase.Atoms): The original atomic structure.
    original_type (str): The type of atom to be replaced.
    dopant (str): The type of dopant atom to introduce.
    num_dopants (int): The number of atoms to replace with the dopant.
    num_structures (int): The number of doped structures to generate.

    Returns:
    list of ase.Atoms: The modified structures with the dopant atom.
    """
    # Get the indices of the atoms to be replaced
    indices_to_replace = [i for i, symbol in enumerate(structure.get_chemical_symbols()) if symbol == original_type]

    # Check if the number of dopants exceeds the available positions
    if num_dopants > len(indices_to_replace):
        raise ValueError(f"Number of dopants ({num_dopants}) exceeds available positions ({len(indices_to_replace)})")

    # Generate all combinations of indices for doping
    all_combinations = list(combinations(indices_to_replace, num_dopants))
    
    # If the number of requested structures exceeds the number of possible combinations, inform the user
    if num_structures > len(all_combinations):
        print(f"Requested number of structures ({num_structures}) exceeds the number of possible unique combinations ({len(all_combinations)}).")
        num_structures = len(all_combinations)

    # Randomly sample the specified number of structures from all possible combinations
    sampled_combinations = random.sample(all_combinations, num_structures)
    
    # Create a list to store the doped structures
    doped_structures = []
    
    for indices in sampled_combinations:
        # Create a copy of the structure
        doped_structure = structure.copy()
        # Replace the atoms at the indices with the dopant
        new_symbols = list(doped_structure.get_chemical_symbols())
        for index in indices:
            new_symbols[index] = dopant
        doped_structure.set_chemical_symbols(new_symbols)
        doped_structures.append(doped_structure)
    
    return doped_structures

def generate_doped_structures(poscar_file, original_type, dopant, num_dopants, num_structures, output_dir):
    """
    Generate new POSCAR files for each combination where the original atom type is replaced by the dopant.

    Parameters:
    poscar_file (str): Path to the original POSCAR file.
    original_type (str): The type of atom to be replaced.
    dopant (str): The type of dopant atom to introduce.
    num_dopants (int): The number of dopants to introduce in each structure.
    num_structures (int): The number of doped structures to generate.
    output_dir (str): Directory to save the new POSCAR files.
    """
    # Read the original structure from the POSCAR file
    structure = read(poscar_file, format='vasp')
    
    # Generate doped structures
    doped_structures = introduce_dopant(structure, original_type, dopant, num_dopants, num_structures)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Write each doped structure to a new POSCAR file in the specified directory
    for i, doped_structure in enumerate(doped_structures):
        output_file = os.path.join(output_dir, f"POSCAR_doped_{i+1}.vasp")
        write(output_file, doped_structure, format='vasp', sort=True)
        print(f"Created {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate doped structures by replacing atoms in a POSCAR file.')
    parser.add_argument('poscar_file', type=str, help='Path to the original POSCAR file')
    parser.add_argument('original_type', type=str, help='The type of atom to be replaced')
    parser.add_argument('dopant', type=str, help='The type of dopant atom to introduce')
    parser.add_argument('num_dopants', type=int, help='The number of dopants to introduce')
    parser.add_argument('num_structures', type=int, help='The number of structures to generate')
    parser.add_argument('output_dir', type=str, help='Directory to save the new POSCAR files')

    args = parser.parse_args()

    generate_doped_structures(args.poscar_file, args.original_type, args.dopant, args.num_dopants, args.num_structures, args.output_dir)

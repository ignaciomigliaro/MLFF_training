import numpy as np
from ase.io import read, write
import random
import os
import argparse

from ase.spacegroup import get_spacegroup

def generate_doped_structure_no_symmetry(structure, original_type, dopant, num_dopants, existing_configurations):
    """
    Generate a single unique doped structure by randomly replacing a specified number of atoms,
    ensuring that symmetry constraints do not interfere.

    Parameters:
    structure (ase.Atoms): The original atomic structure.
    original_type (str): The type of atom to be replaced.
    dopant (str): The type of dopant atom to introduce.
    num_dopants (int): The number of atoms to replace with the dopant.
    existing_configurations (set): A set of already generated configurations to avoid duplicates.

    Returns:
    ase.Atoms: The modified structure with the dopant atom.
    """
    # Remove symmetry constraints (if any)
    structure_no_symmetry = structure.copy()
    structure_no_symmetry.set_pbc((True, True, True))  # Ensure periodic boundary conditions
    spacegroup = get_spacegroup(structure_no_symmetry)
    structure_no_symmetry = spacegroup.no_symmetry(structure_no_symmetry)  # Remove symmetry
    
    # Get the indices of the atoms to be replaced
    indices_to_replace = [i for i, symbol in enumerate(structure_no_symmetry.get_chemical_symbols()) if symbol == original_type]

    # Check if the number of dopants exceeds the available positions
    if num_dopants > len(indices_to_replace):
        raise ValueError(f"Number of dopants ({num_dopants}) exceeds available positions ({len(indices_to_replace)})")

    # Randomly select unique indices for doping until a new unique configuration is found
    while True:
        selected_indices = tuple(sorted(random.sample(indices_to_replace, num_dopants)))
        if selected_indices not in existing_configurations:
            existing_configurations.add(selected_indices)
            break

    # Create a copy of the structure
    doped_structure = structure_no_symmetry.copy()
    
    # Replace the atoms at the selected indices with the dopant
    for index in selected_indices:
        doped_structure[index].symbol = dopant  # Directly replace the atom's symbol
    
    return doped_structure


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
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Track already generated configurations to avoid duplicates
    existing_configurations = set()
    
    # Generate and write each doped structure to a new POSCAR file in its own directory
    for i in range(num_structures):
        # Generate a unique doped structure
        doped_structure = generate_doped_structure(structure, original_type, dopant, num_dopants, existing_configurations)
        
        # Create a directory for each structure
        structure_dir = os.path.join(output_dir, f"structure_{num_dopants}_dopes_{i+1}")
        os.makedirs(structure_dir, exist_ok=True)
        
        # Write the POSCAR file inside the new directory
        output_file = os.path.join(structure_dir, "POSCAR")
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

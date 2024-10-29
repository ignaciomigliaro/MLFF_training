import numpy as np
from ase.io import read, write
import random
import os
import argparse

def generate_doped_structure(structure, original_type, dopant, num_dopants, existing_configurations):
    """
    Generate a single unique doped structure by randomly replacing a specified number of atoms.

    Parameters:
    structure (ase.Atoms): The original atomic structure.
    original_type (str): The type of atom to be replaced.
    dopant (str): The type of dopant atom to introduce.
    num_dopants (int): The number of atoms to replace with the dopant.
    existing_configurations (set): A set of already generated configurations to avoid duplicates.

    Returns:
    ase.Atoms: The modified structure with the dopant atom.
    """
    # Get the indices of the atoms to be replaced
    indices_to_replace = [i for i, symbol in enumerate(structure.get_chemical_symbols()) if symbol == original_type]

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
    doped_structure = structure.copy()

    # Replace the atoms at the selected indices with the dopant
    for index in selected_indices:
        doped_structure[index].symbol = dopant

    return doped_structure

def write_qe_file(output_directory, crystal_structure):
    """
    Write a Quantum Espresso input file using ASE's write function.

    Parameters:
    output_directory (str): Path to save the Quantum Espresso input file.
    crystal_structure (ase.Atoms): The atomic structure to write.
    """
    # Quantum Espresso calculation parameters
    input_data = {
        "calculation": "vc-relax",
        "prefix": "qe_input",
        "pseudo_dir": "/home/dal950773/QE/pseudo",
        "outdir": "./out/",
        "verbosity": "high",
        "etot_conv_thr": 1.0e-03,
        "forc_conv_thr": 5.0e-02,
        "tstress": True,
        "degauss": 1.4699723600e-02,
        "ecutrho": 200,
        "ecutwfc": 50,
        "vdw_corr": "mbd",
        "occupations": "smearing",
        "smearing": 'cold',
        "electron_maxstep": 80,
        "mixing_beta": 4.0e-01,
        }
    # Pseudopotentials for different elements
    pseudos = {
        'Cl': 'Cl.upf',
        'O': 'O.upf',
        'F': 'F.upf',
        'I': 'I.upf',
        'Br': 'Br.upf',
        'La': 'La.upf',
        'Li': 'Li.upf',
        'Zr': 'Zr.upf',
        'C' : 'C.upf',
        'H': 'H.upf'
    }

    # Use ASE's write function to generate the Quantum Espresso input file
    write(
        filename=output_directory,
        images=crystal_structure,
        format='espresso-in',
        input_data=input_data,
        pseudopotentials=pseudos,
        kspacing=0.05,
    )

def generate_doped_structures(poscar_file, original_type, dopant, num_dopants, num_structures, output_dir, qe=False):
    """
    Generate new structures for each combination where the original atom type is replaced by the dopant.
    Conditionally write either VASP POSCAR files or Quantum Espresso input files.

    Parameters:
    poscar_file (str): Path to the original POSCAR file.
    original_type (str): The type of atom to be replaced.
    dopant (str): The type of dopant atom to introduce.
    num_dopants (int): The number of dopants to introduce in each structure.
    num_structures (int): The number of doped structures to generate.
    output_dir (str): Directory to save the new structure files.
    qe (bool): If True, generate Quantum Espresso input files instead of VASP POSCAR files.

    Returns:
    list: A list of ASE Atoms objects representing the generated doped structures.
    """
    # Read the original structure from the POSCAR file
    structure = read(poscar_file, format='vasp')
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Track already generated configurations to avoid duplicates
    existing_configurations = set()
    
    # List to store the generated structures
    generated_structures = []
    
    # Generate and write each doped structure
    for i in range(num_structures):
        # Generate a unique doped structure
        doped_structure = generate_doped_structure(structure, original_type, dopant, num_dopants, existing_configurations)
        
        # Save the doped structure in the list
        generated_structures.append(doped_structure)
        
        # Create a directory for each structure
        structure_dir = os.path.join(output_dir, f"structure_{num_dopants}_dopes_{i+1}")
        os.makedirs(structure_dir, exist_ok=True)
        
        if qe:
            # Write the Quantum Espresso file
            output_file = os.path.join(structure_dir, "pwscf.in")
            write_qe_file(output_file, doped_structure)
            print(f"Created Quantum Espresso input file: {output_file}")
        else:
            # Write the VASP POSCAR file
            output_file = os.path.join(structure_dir, "POSCAR")
            write(output_file, doped_structure, format='vasp', sort=True)
            print(f"Created VASP POSCAR file: {output_file}")

    return generated_structures  # Return the list of generated structures



def main():
    parser = argparse.ArgumentParser(description="Generate doped structures and optionally write Quantum Espresso input files.")
    parser.add_argument('poscar_file', type=str, help='Path to the input POSCAR file.')
    parser.add_argument('original_type', type=str, help='Type of atom to be replaced (e.g., Nb).')
    parser.add_argument('dopant', type=str, help='Type of dopant atom (e.g., C).')
    parser.add_argument('num_dopants', type=int, help='Number of dopants to add.')
    parser.add_argument('num_structures', type=int, help='Number of configurations to generate.')
    parser.add_argument('output_dir', type=str, help='Output directory for the generated structures.')
    parser.add_argument('--qe', action='store_true', help='Generate a Quantum Espresso input file for each doped structure.')

    args = parser.parse_args()

    # Generate doped structures with the option to write QE files
    generate_doped_structures(
        args.poscar_file, 
        args.original_type, 
        args.dopant, 
        args.num_dopants, 
        args.num_structures, 
        args.output_dir,
        qe=args.qe
    )

if __name__ == '__main__':
    main()
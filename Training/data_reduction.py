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
import pickle
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(
        description="This is a code to parse for Chgnet and have fast parsing"
    )
    parser.add_argument(
        "input_filepath",
        help="Directory that contains all of your VASP files"
    )
    parser.add_argument(
         '--verbose',
         action='store_true',
         help="If you want to see any error occurring in the parsing process"
    )
    return parser.parse_args()

def parse_vasp_dir(filepath, verbose, stepsize=1):
    """This is a function to replace the Chgnet Utils, this is more robust and gives correct energy values"""
    warnings.filterwarnings('ignore')
    atoms_list = []
    len_list = []
    total_iterations = len(os.listdir(filepath))

    with tqdm(total=total_iterations, desc='Processing') as pbar:
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
                        e = a.get_total_energy()
                        a.info['relaxed_energy'] = last_energy
            except Exception as e:
                if verbose:
                    print(f"Error reading file: {file_path}")
                    print(f"Error details: {e}")
                continue
            finally:
                pbar.update(1)

    return atoms_list

#TODO def random_sample(atoms_list,percentage=2):

def write_mace(output,atoms_list):
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

#TODO write_yaml()

#TODO write_slurm()

#TODO def train_mace(atom_list)
#call write_yaml
#call write_slurm 
#submits slurm
#returns slurm ID

#TODO check if finished

#TODO inference_mace()

#TODO def calculate_error(dft,mlff):

#TODO def save_data(threshold):

#TODO def evaluate():
#determines how many failing cases. 

def main():
    args = parse_args()
    input = Path(args.input_filepath)
    atoms_list = parse_vasp_dir(input)


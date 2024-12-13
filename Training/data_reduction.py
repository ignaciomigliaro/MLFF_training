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
import random
import yaml
import subprocess
import time
import threading
from mace.calculators import MACECalculator



def parse_args():
    parser = argparse.ArgumentParser(
        description="This is a code to parse for MACE and have fast parsing"
    )
    parser.add_argument(
        "input_filepath",
        help="Directory that contains all of your VASP files"
    )
    parser.add_argument(
        '--sampling_percentage',
        type=int,
        default=100,
        help="Percentage of configurations to sample (0-100)"
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="If you want to see any error occurring in the parsing process"
    )
    parser.add_argument(
        '--slurm_output',
        default="train_slurm",
        help="Output path for the generated SLURM script"
    )
    parser.add_argument(
        '--job_name',
        default="LLZO",
        help="Name of the SLURM job"
    )
    return parser.parse_args()

def parse_vasp_dir(filepath, verbose):
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
                single_file_atom = read(file_path, index=':')
                all_steps = list(single_file_atom)
                len_list.append(len(all_steps))
                
                for a in all_steps:
                    energy = a.get_total_energy()
                    if energy < 0:
                        a.info['file'] = filepath.joinpath(i)
                        a.info['relaxed_energy'] = energy
                        atoms_list.append(a)
            except Exception as e:
                if verbose:
                    print(f"Error reading file: {file_path}")
                    print(f"Error details: {e}")
                continue
            finally:
                pbar.update(1)

    return atoms_list

def random_sampling(atoms_list, percentage):
    if not (0 <= percentage <= 100):
        raise ValueError("Percentage must be between 0 and 100")
    num_to_sample = int(len(atoms_list) * (percentage / 100))
    sampled_atoms = random.sample(atoms_list, num_to_sample)
    remaining_atoms = [atom for atom in atoms_list if atom not in sampled_atoms]
    return sampled_atoms, remaining_atoms

def write_mace(output, atoms_list):
    train_data, test_data = train_test_split(atoms_list, test_size=0.1, random_state=42)
    print(f"Total data: {len(atoms_list)}, Training data: {len(train_data)}, Testing data: {len(test_data)}")

    base = str(output).rsplit('.', 1)[0] if '.' in str(output) else str(output)
    train_file = f"{base}_train.xyz"
    test_file = f"{base}_test.xyz"

    write(train_file, train_data)
    write(test_file, test_data)

def generate_yaml_config(output_base):
    train_file = f"{output_base}_train.xyz"
    test_file = f"{output_base}_test.xyz"
    config_dict = {
        "name": "LLZO",
        "model_dir": "MACE_models",  # Directory where the model will be saved
        "log_dir": "MACE_models",  # Log directory for the training process
        "checkpoints_dir": "MACE_models",  # Checkpoints for model saving
        "train_file": train_file,
        "test_file": test_file,
        "swa": True,
        "max_num_epochs": 20,
        "batch_size": 10,
        "device": "cuda",  # Device for training (can be 'cpu' or 'cuda')
        "E0s": "average",
        "valid_fraction": 0.05,
        "save_cpu": True,
    }
    return config_dict

def write_yaml_file(output_path, config_dict):
    with open(output_path, 'w') as yaml_file:
        yaml.dump(config_dict, yaml_file, default_flow_style=False)


def write_slurm_script(output_path, yaml_config_path, job_name="LLZO"):
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=100gb
#SBATCH -p share.gpu

module load cuda11.8/toolkit/11.8.0
module load MINICONDA/23.1.0_Py3.9

source /cm/shared/apps/MINICONDA/23.1.0_Py3.9/etc/profile.d/conda.sh

conda activate mace2
python -m mace.cli.run_train --config="{yaml_config_path}"
"""
    with open(output_path, "w") as file:
        file.write(slurm_script)

def generate_and_submit(yaml_config_path, slurm_script_path, job_name="LLZO"):
    print("Generating YAML configuration...")
    yaml_config = generate_yaml_config("mace")
    write_yaml_file(yaml_config_path, yaml_config)
    print(f"YAML configuration written to: {yaml_config_path}")

    print("Writing SLURM script...")
    write_slurm_script(slurm_script_path, yaml_config_path, job_name)
    print(f"SLURM script written to: {slurm_script_path}")

    print("Submitting SLURM job...")
    submit_command = f"sbatch {slurm_script_path}"
    try:
        result = subprocess.run(submit_command, shell=True, check=True, capture_output=True, text=True)
        slurm_job_id = result.stdout.split()[-1]
        print(f"SLURM job submitted successfully with job ID: {slurm_job_id}")
        return slurm_job_id
    except subprocess.CalledProcessError as e:
        print(f"Error submitting SLURM job: {e}")
        return None

def is_job_finished(job_id):
    """
    Check if a SLURM job with a given job ID has finished.
    """
    try:
        username = subprocess.check_output("whoami", text=True).strip()
        command = f"squeue -u {username} -h -o %i"  # Check if job ID exists in the queue
        output = subprocess.check_output(command, shell=True, text=True)
        job_ids = output.splitlines()
        if job_id in job_ids:
            return False  # Job is still running
        return True  # Job is finished
    except subprocess.CalledProcessError as e:
        print(f"Error checking job status: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def monitor_slurm_job(slurm_job_id, yaml_config):
    """
    Monitor the SLURM job and construct the model path using fixed components from the YAML config.
    """
    while not is_job_finished(slurm_job_id):
        print(f"SLURM job {slurm_job_id} is still running. Checking again in 60 seconds...")
        time.sleep(30)  # Check every 30 seconds

    print(f"SLURM job {slurm_job_id} has finished.")

    # Extract model information from YAML config
    model_name = yaml_config.get("name", "")
    model_dir = yaml_config.get("model_dir", "")

    if not model_name or not model_dir:
        print("Error: 'name' or 'model_dir' is not defined in the YAML config.")
        return None

    # Construct model path
    model_path = f"{model_dir}/{model_name}_swa.model"

    if Path(model_path).exists():
        print(f"Constructed model path: {model_path}")
        return model_path
    else:
        print(f"Error: Model path does not exist: {model_path}")
        return None

def calculate_total_energy(atoms_list, model_path):
    """
    Calculates total energy and forces for each configuration using the MACE model.
    """
    # Initialize MACE calculator
    try:
        calc = MACECalculator(model_paths=model_path, device='cpu')
    except Exception as e:
        print(f"Error initializing MACE calculator: {e}")
        return None
    
    # Create a deep copy of the atoms list to avoid modifying the original atoms
    copied_atoms = [atom.copy() for atom in atoms_list]
    
    energies = []
    forces = []
    for atom in copied_atoms:
        atom.set_calculator(calc)  # Attach the calculator to the atom
        
        # Calculate total energy for the configuration
        energy = atom.get_total_energy()
        force = atom.get_forces()
        
        # Store the energy and forces in atom's info dictionary
        atom.info['mace_energy'] = energy  
        atom.info['mace_forces'] = force
        
        energies.append(energy)  # Store energy of each configuration
        forces.append(force)    # Store forces of each configuration
    
    # Return the updated list of atoms along with energy and force information
    return copied_atoms, energies, forces

import numpy as np

def calculate_energy_error(remaining_atoms, model_path, output_file="energy_errors.csv"):
    """
    Calculate the error between VASP energies and MACE-predicted energies for remaining atoms
    and write the results to a file.
    
    Parameters:
    - remaining_atoms: List of ASE Atoms objects with VASP energies in `relaxed_energy`.
    - model_path: Path to the trained MACE model.
    - output_file: File to save energies and errors.
    
    Returns:
    - errors: List of absolute errors for each atom.
    - mae: Mean Absolute Error.
    - rmse: Root Mean Square Error.
    """
    # Calculate MACE energies
    updated_atoms, mace_energies, _ = calculate_total_energy(remaining_atoms, model_path)
    
    # Extract VASP energies
    vasp_energies = [atom.info['relaxed_energy'] for atom in remaining_atoms]
    
    # Calculate errors
    errors = [abs(vasp - mace) for vasp, mace in zip(vasp_energies, mace_energies)]
    mae = np.mean(errors)  # Mean Absolute Error
    rmse = np.sqrt(np.mean(np.square(errors)))  # Root Mean Square Error
    
    # Write to file
    with open(output_file, "w") as f:
        f.write("Index,VASP_Energy(eV),MACE_Energy(eV),Error(eV)\n")
        for idx, (vasp, mace, error) in enumerate(zip(vasp_energies, mace_energies, errors), start=1):
            f.write(f"{idx},{vasp:.6f},{mace:.6f},{error:.6f}\n")
        f.write(f"\nMean Absolute Error (MAE):,{mae:.6f} eV\n")
        f.write(f"Root Mean Square Error (RMSE):,{rmse:.6f} eV\n")
    
    print(f"Energies and errors written to {output_file}")
    return errors, mae, rmse

def main():
    args = parse_args()

    input_filepath = Path(args.input_filepath)
    if not input_filepath.exists() or not input_filepath.is_dir():
        raise FileNotFoundError(f"The input filepath '{input_filepath}' is not a valid directory.")

    atoms_list = parse_vasp_dir(input_filepath, verbose=args.verbose)

    if not atoms_list:
        print("No valid configurations found.")
        return

    # Perform random sampling
    sampled_atoms, remaining_atoms = random_sampling(atoms_list, args.sampling_percentage)

    print(f"Randomly sampled {len(sampled_atoms)} configurations:")
    for idx, atom in enumerate(sampled_atoms, start=1):
        energy = atom.info.get('relaxed_energy', 'N/A')
        print(f"Configuration {idx}: Energy = {energy:.6f} eV")

    base_output = "mace"
    write_mace(base_output, sampled_atoms)

    yaml_config_path = f"{base_output}_config.yaml"  # Name of the generated YAML config file
    slurm_script_path = args.slurm_output

    # Generate the YAML config and write it to file
    config_dict = generate_yaml_config(base_output)
    write_yaml_file(yaml_config_path, config_dict)

    # Submit SLURM job and get the job ID
    slurm_job_id = generate_and_submit(yaml_config_path, slurm_script_path, args.job_name)

    if slurm_job_id:
        print(f"Submitted SLURM job with ID {slurm_job_id}. Monitoring for completion...")
        model_path = monitor_slurm_job(slurm_job_id, config_dict)  # Pass YAML config here

        # Calculate errors and write to file
        if model_path:
            output_file = "energy_errors.csv"
            errors, mae, rmse = calculate_energy_error(remaining_atoms, model_path, output_file)
        
            print(f"Calculated energy errors for {len(errors)} configurations.")
            print(f"Mean Absolute Error (MAE): {mae:.6f} eV")
            print(f"Root Mean Square Error (RMSE): {rmse:.6f} eV")
        else:
            print("Failed to locate trained MACE model. Cannot calculate errors.")
    else:
        print("Failed to submit SLURM job.")

if __name__ == "__main__":
    main()
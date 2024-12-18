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
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
        default=50,
        help="Percentage of configurations to sample (0-100)"
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="If you want to see any error occurring in the parsing process"
    )
    parser.add_argument(
        '--job_name',
        default="LLZO",
        help="Name of the SLURM job"
    )
    parser.add_argument(
    '--threshold',
    type=float,
    default=10.0,
    help="Threshold for terminating the data reduction training loop"
    )
    parser.add_argument(
    '--device',
    type=str,
    default='cpu',
    help="Device to run training and inference runs (Default = 'cpu')"
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
                    logging.info(f"No valid file ('OUTCAR' or 'pwscf.out') found in {filepath / i}")
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
                    logging.info(f"Error reading file: {file_path}")
                    logging.info(f"Error details: {e}")
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
    logging.info(f"Total data: {len(atoms_list)}, Training data: {len(train_data)}, Testing data: {len(test_data)}")

    base = str(output).rsplit('.', 1)[0] if '.' in str(output) else str(output)
    train_file = f"{base}_train.xyz"
    test_file = f"{base}_test.xyz"

    write(train_file, train_data)
    write(test_file, test_data)

def submit_job(yaml_config_path, slurm_script_path):
    """
    Submit a SLURM job using a provided YAML configuration and SLURM script.
    """
    # Check if the files exist
    if not os.path.exists(yaml_config_path):
        logging.error(f"Error: YAML configuration file '{yaml_config_path}' not found.")
        return None
    if not os.path.exists(slurm_script_path):
        logging.error(f"Error: SLURM script file '{slurm_script_path}' not found.")
        return None

    logging.info(f"Using YAML configuration file: {yaml_config_path}")
    logging.info(f"Using SLURM script file: {slurm_script_path}")

    # Submit the SLURM job
    logging.info("Submitting SLURM job...")
    submit_command = f"sbatch {slurm_script_path}"
    try:
        result = subprocess.run(submit_command, shell=True, check=True, capture_output=True, text=True)
        slurm_job_id = result.stdout.split()[-1]
        logging.info(f"SLURM job submitted successfully with job ID: {slurm_job_id}")
        return slurm_job_id
    except subprocess.CalledProcessError as e:
        logging.error(f"Error submitting SLURM job: {e}")
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
        logging.error(f"Error checking job status: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return False

def read_yaml_file(yaml_config_path):
    """
    Reads the YAML configuration file.
    """
    if not os.path.exists(yaml_config_path):
        logging.error(f"Error: YAML configuration file '{yaml_config_path}' not found.")
        return None
    try:
        with open(yaml_config_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        logging.info(f"YAML configuration loaded successfully from: {yaml_config_path}")
        return config
    except Exception as e:
        logging.info(f"Error reading YAML configuration file: {e}")
        return None

def monitor_slurm_job(slurm_job_id, yaml_config):
    """
    Monitor the SLURM job and construct the model path using fixed components from the YAML config.
    """
    while not is_job_finished(slurm_job_id):
        logging.info(f"SLURM job {slurm_job_id} is still running. Checking again in 60 seconds...")
        time.sleep(30)  # Check every 30 seconds

    logging.info(f"SLURM job {slurm_job_id} has finished.")

    # Extract model information from YAML config
    model_name = yaml_config.get("name", "")
    model_dir = yaml_config.get("model_dir", "")

    if not model_name or not model_dir:
        logging.error("Error: 'name' or 'model_dir' is not defined in the YAML config.")
        return None

    # Construct model path
    model_path = f"{model_dir}/{model_name}_swa.model"

    if Path(model_path).exists():
        logging.info(f"Constructed model path: {model_path}")
        return model_path
    else:
        logging.error(f"Error: Model path does not exist: {model_path}")
        return None

def calculate_total_energy(atoms_list, model_path,device='cpu'):
    """
    Calculates total energy and forces for each configuration using the MACE model.
    """
    # Initialize MACE calculator
    try:
        calc = MACECalculator(model_paths=model_path, device=device)
    except Exception as e:
        logging.error(f"Error initializing MACE calculator: {e}")
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

def calculate_energy_error(remaining_atoms, model_path,iteration,device='cpu', output_file="energy_errors_per_atom.csv"):
    """
    Calculate the per-atom error between VASP energies and MACE-predicted energies for remaining atoms
    and write the results to a file.
    
    Parameters:
    - remaining_atoms: List of ASE Atoms objects with VASP energies in `relaxed_energy`.
    - model_path: Path to the trained MACE model.
    - output_file: File to save energies and errors.
    
    Returns:
    - errors_per_atom: List of per-atom errors for each system.
    - mae_per_atom: Mean Absolute Error per atom.
    - rmse_per_atom: Root Mean Square Error per atom.
    """
    # Calculate MACE energies
    updated_atoms, mace_energies, _ = calculate_total_energy(remaining_atoms, model_path,device)
    
    # Extract VASP energies and number of atoms
    vasp_energies = [atom.info['mace_energy'] for atom in remaining_atoms]
    num_atoms = [len(atom) for atom in remaining_atoms]
    
    # Normalize energies by number of atoms
    vasp_energies_per_atom = [vasp / n for vasp, n in zip(vasp_energies, num_atoms)]
    mace_energies_per_atom = [mace / n for mace, n in zip(mace_energies, num_atoms)]
    
    # Calculate per-atom errors
    errors_per_atom = [abs(vasp - mace) for vasp, mace in zip(vasp_energies_per_atom, mace_energies_per_atom)]
    mae_per_atom = np.mean(errors_per_atom)  # Mean Absolute Error per atom
    rmse_per_atom = np.sqrt(np.mean(np.square(errors_per_atom)))  # Root Mean Square Error per atom
    
    # Write to file
    with open(output_file, "w") as f:
        f.write("Index,Iteration,VASP_Energy_per_Atom(eV),MACE_Energy_per_Atom(eV),Error_per_Atom(eV)\n")
        for idx, (vasp, mace, error) in enumerate(zip(vasp_energies_per_atom, mace_energies_per_atom, errors_per_atom), start=1):
            f.write(f"{idx},{iteration},{vasp:.6f},{mace:.6f},{error:.6f}\n")
        f.write(f"\nMean Absolute Error per Atom (MAE):,{mae_per_atom:.6f} eV\n")
        f.write(f"Root Mean Square Error per Atom (RMSE):,{rmse_per_atom:.6f} eV\n")
    
    logging.info(f"Energies and errors per atom written to {output_file}")
    return errors_per_atom, mae_per_atom, rmse_per_atom

def filter_and_resample_failed_cases(errors_per_atom, remaining_atoms, sampled_atoms, threshold=0.04254, resample_percentage=30):
    """
    Filters and resamples failed cases based on a threshold error.

    Parameters:
        errors_per_atom (list): List of energy errors per atom.
        remaining_atoms (list): List of remaining atoms.
        sampled_atoms (list): List of sampled atoms.
        threshold (float): Threshold to identify failed cases.
        resample_percentage (float): Percentage of failed cases to resample.

    Returns:
        tuple: Updated sampled_atoms and remaining_atoms.
    """
    # Identify failed cases
    failed_indices = [i for i, error in enumerate(errors_per_atom) if error > threshold]
    failed_cases = [remaining_atoms[i] for i in failed_indices]

    # Calculate percentage of failed cases
    if len(errors_per_atom) > 0:
        failed_percentage = (len(failed_cases) / len(errors_per_atom)) * 100
        logging.info(f"Percentage of failed cases: {failed_percentage:.2f}%")
    else:
        logging.info("No errors to process.")
        return sampled_atoms, remaining_atoms, 0.0

    # If failed_percentage is less than 10, add all failed cases to sampled_atoms
    if failed_percentage < 10.0:
        logging.info(f"Failed percentage is less than 10%, adding all failed cases to sampled_atoms.")
        sampled_atoms.extend(failed_cases)
        remaining_atoms = [atom for atom in remaining_atoms if atom not in failed_cases]
        failed_percentage = 0.0  # No need to resample further
    else:
        # Resample 30% of failed cases
        num_to_resample = int(len(failed_cases) * (resample_percentage / 100))
        resampled_cases = random.sample(failed_cases, num_to_resample)

        # Update sampled atoms
        sampled_atoms.extend(resampled_cases)

        # Remaining atoms include non-failed cases and failed cases not resampled
        remaining_atoms = [atom for i, atom in enumerate(remaining_atoms) if i not in failed_indices or remaining_atoms[i] not in resampled_cases]

        logging.info(f"Resampled {len(resampled_cases)} failed cases and added them to sampled atoms.")

    return sampled_atoms, remaining_atoms, failed_percentage

def main():
    args = parse_args()

    input_filepath = Path(args.input_filepath)
    if not input_filepath.exists() or not input_filepath.is_dir():
        raise FileNotFoundError(f"The input filepath '{input_filepath}' is not a valid directory.")

    atoms_list = parse_vasp_dir(input_filepath, verbose=args.verbose)

    if not atoms_list:
        logging.error("No valid configurations found.")
        return

    # Perform random sampling
    sampled_atoms, remaining_atoms = random_sampling(atoms_list, args.sampling_percentage)

    logging.info(f"Randomly sampled {len(sampled_atoms)} configurations:")
    for idx, atom in enumerate(sampled_atoms, start=1):
        energy = atom.info.get('relaxed_energy', 'N/A')
        if args.verbose:
            logging.info(f"Configuration {idx}: Energy = {energy:.6f} eV")

    iteration = 1
    failed_percentage = 100.0  # Initialize failed_percentage
    threshold = args.threshold
    while failed_percentage > threshold:
        logging.info(f"\nIteration {iteration}: Submitting job with current training data...")

        base_output = "mace"
        write_mace(base_output, sampled_atoms)

        yaml_config_path = f"{base_output}_config.yaml"  # Name of the generated YAML config file
        slurm_script_path = f"{base_output}.slurm"

        # Submit SLURM job and get the job ID
        slurm_job_id = submit_job(yaml_config_path, slurm_script_path)

        if slurm_job_id:
            logging.info(f"Submitted SLURM job with ID {slurm_job_id}. Monitoring for completion...")
            config_dict = read_yaml_file(yaml_config_path)
            model_path = monitor_slurm_job(slurm_job_id, config_dict)  # Pass YAML config here

            # Calculate errors and write to file
            if model_path:
                output_file = f"energy_errors_iter.csv"
                errors, mae, rmse = calculate_energy_error(remaining_atoms, model_path,iteration,args.device,output_file)
            
                logging.info(f"Calculated energy errors for {len(errors)} configurations.")
                logging.info(f"Mean Absolute Error (MAE): {mae:.6f} eV")
                logging.info(f"Root Mean Square Error (RMSE): {rmse:.6f} eV")

                # Filter and resample failed cases
                sampled_atoms, remaining_atoms, failed_percentage = filter_and_resample_failed_cases(
                    errors, remaining_atoms, sampled_atoms
                )
                logging.info(f"Filtered and resampled failed cases. Failed percentage: {failed_percentage:.2f}%")
            else:
                logging.error("Failed to locate trained MACE model. Cannot calculate errors.")
                return
        else:
            logging.error("Failed to submit SLURM job.")
            return

        iteration += 1

    logging.info("\nTraining process completed. Final dataset prepared.")
    write_mace("final_output", sampled_atoms)
    logging.info("Final training data written to final_output.")

if __name__ == "__main__":
    main()

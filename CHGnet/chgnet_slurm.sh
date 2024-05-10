#!/bin/bash
#SBATCH --job-name=chgnet_AIMD
#SBATCH -o slurm-%j.out            # Output file
#SBATCH -e slurm-%j.err            # Error file
#SBATCH --ntasks-per-node=8        # Number of tasks (or cores)
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                  # Number of nodes to be used
#SBATCH --mem=100gb                 # Adjust based on actual memory requirements
#SBATCH -p share.gpu                # Partition or queue name

module load cuda11.8/toolkit/11.8.0
module load MINICONDA/23.1.0_Py3.9
cd <work_dir>
python3 -u /storage/nas_scr/im0225/NbOC/chgnet/debug.py <pickle_file>

#!/usr/bin/bash
# Slurm bash script

#SBATCH --job-name=otis_base
#SBATCH --output=/vol/aimspace/users/tuo/otis/slurm_output/pre/otis_base-%A.out   # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=/vol/aimspace/users/tuo/otis/slurm_output/pre/otis_base-%A.err    # Standard error of the script
#SBATCH --time=6-23:59:59   # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:A40:2   # Number of GPUs if needed
#SBATCH --gpu-bind=none     # NCCL can't deal with task-binding (binding each process to its own GPU), hence allocate the devices manually in the python script
#SBATCH --cpus-per-gpu=32   # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --mem-per-gpu=128G  # Memory in GB (Don't use more than 128G per GPU)

# load python module
module load python/anaconda3
source /opt/anaconda3/etc/profile.d/conda.sh

# activate corresponding environment
conda deactivate            # If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. This guards against that. Not necessary if you always run this script from a clean terminal
conda activate mae4

cmd="./bash_scripts/pretrain/pretrain_otis_base_1p.sh 31"
echo $cmd && $cmd
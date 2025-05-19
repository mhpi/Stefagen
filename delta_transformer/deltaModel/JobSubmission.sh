#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=240GB 
#SBATCH -C v100s
#SBATCH --gpus=2
#SBATCH --time=40:00:00
#SBATCH --partition=mgc-mri
#SBATCH --account=cxs1024_mri
#SBATCH --job-name=DMG531RmseLoss5HBV1.1
#SBATCH --output=%x_%j.out

# Load necessary modules
module purge
module load parallel
module load anaconda3

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate the MFFormer environment
conda activate deltamodel
cd ..


# Add the parent directory of MFFormer to PYTHONPATH
export PYTHONPATH="$PWD:$PYTHONPATH"

python /storage/home/nrk5343/work/generic_deltaModel_withTransformer/deltaModel/finetune_main.py

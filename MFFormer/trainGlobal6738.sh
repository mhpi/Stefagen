#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=4
#SBATCH --mem=360GB 
#SBATCH -C v100s
#SBATCH --gpus=6
#SBATCH --time=170:00:00
#SBATCH --partition=mgc-mri
#SBATCH --account=cxs1024_mri
#SBATCH --job-name=FoundationalModelTrainGlobalFull
#SBATCH --output=%x_%j.out

# Load necessary modules
module purge
module load parallel
module load anaconda3

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate the MFFormer environment
conda activate MFFormerTrain



# Add the parent directory of MFFormer to PYTHONPATH
export PYTHONPATH="$PWD:$PYTHONPATH"

python3 -u MFFormer/run2.py \
    --task_name pretrain \
    --model MFFormer_dec_LSTM \
    --data  GLOBAL_6738 \
    --time_series_variables P,Temp,PET,streamflow \
    --train_date_list 1980-01-01,2016-12-31 \
    --val_date_list 1990-01-01,2010-12-31 \
    --test_date_list 1980-01-01,2016-12-31 \
    --mask_ratio_time_series 0.5 \
    --mask_ratio_static 0.5 \
    --num_heads 4 \
    --num_enc_layers 4 \
    --num_dec_layers 2 \
    --dropout 0.1 \
    --epochs 15 \
    --batch_size 516 \
    --learning_rate 0.0001 \
    --clip_grad 1.0 \
    --calculate_time_series_each_variable_loss \
    --use_gpu \
    --gpu 0 \
    --use_multi_gpu \
    --devices 0,1,2,3,4,5 \
    --num_kfold 5 \
    --do_eval \
    --nth_kfold 0 \
    --station_ids_file GLOBAL_6738 \
    --des Global6738Full \

conda deactivate
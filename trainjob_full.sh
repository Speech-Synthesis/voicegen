#!/bin/bash
#PBS -N hdvc_full
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb
#PBS -l walltime=72:00:00
#PBS -q gpu
#PBS -o logs/trainjob_full.out
#PBS -e logs/trainjob_full.err

cd $PBS_O_WORKDIR

echo "Loading modules..."
module load cuda11.6/toolkit/11.6.2

echo "Activating micromamba environment..."
eval "$(micromamba shell hook --shell bash)"
micromamba activate voicegen

echo "Starting Full Research Model Training (Stage 9)..."
# Full configuration with MINE/Cosine disentanglement loss
python train_full.py --config configs/vctk_full.json --model_dir checkpoints/full

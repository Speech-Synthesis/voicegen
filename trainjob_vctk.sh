#!/bin/bash
#PBS -N hdvc_baseline_vctk
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb
#PBS -l walltime=48:00:00
#PBS -q gpu
#PBS -o logs/trainjob_vctk.out
#PBS -e logs/trainjob_vctk.err

cd $PBS_O_WORKDIR

echo "Loading modules..."
module load cuda11.6/toolkit/11.6.2

echo "Activating micromamba environment..."
eval "$(micromamba shell hook --shell bash)"
micromamba activate voicegen

echo "Starting Baseline Multi-Speaker Training (Stage 5)..."
# Full VCTK base config for baseline comparison
python train_full.py --config configs/vctk_base.json --model_dir checkpoints/baseline_vctk

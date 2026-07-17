#!/bin/bash
#PBS -N hdvc_libritts
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb
#PBS -l walltime=72:00:00
#PBS -q gpu
#PBS -o logs/trainjob_libritts.out
#PBS -e logs/trainjob_libritts.err

cd $PBS_O_WORKDIR

echo "Loading modules..."
module load cuda11.6/toolkit/11.6.2

echo "Activating micromamba environment..."
eval "$(micromamba shell hook --shell bash)"
micromamba activate voicegen

echo "Starting LibriTTS Multi-Speaker Training..."
# LibriTTS train-clean-100 baseline (247 speakers)
python train_full.py --config configs/libritts_base.json --model_dir checkpoints/baseline_libritts

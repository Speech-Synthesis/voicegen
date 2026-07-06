#!/bin/bash
#PBS -N hdvc_prosody
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb
#PBS -l walltime=24:00:00
#PBS -q gpu
#PBS -o logs/trainjob_prosody.out
#PBS -e logs/trainjob_prosody.err

cd $PBS_O_WORKDIR

echo "Loading modules..."
module load cuda11.6/toolkit/11.6.2

echo "Activating micromamba environment..."
eval "$(micromamba shell hook --shell bash)"
micromamba activate voicegen

echo "Starting Standalone Prosody Pretraining (Stage 7)..."
python train_prosody.py --config configs/vctk_prosody_pretrain.json --model_dir checkpoints/prosody_encoder

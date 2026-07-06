#!/bin/bash
#PBS -N hdvc_fusion
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb
#PBS -l walltime=48:00:00
#PBS -q gpu
#PBS -o logs/trainjob_fusion.out
#PBS -e logs/trainjob_fusion.err

cd $PBS_O_WORKDIR

echo "Loading modules..."
module load cuda11.6/toolkit/11.6.2

echo "Activating micromamba environment..."
eval "$(micromamba shell hook --shell bash)"
micromamba activate voicegen

echo "Starting Joint Fusion Module Training (Stage 8)..."
# Use a config where use_prosody_encoder=True but disentanglement might be disabled
python train_full.py --config configs/vctk_abl_nodis.json --model_dir checkpoints/fusion_nodis

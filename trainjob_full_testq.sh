#!/bin/bash
#PBS -N hdvc_full
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb
#PBS -l walltime=96:00:00
#PBS -q testq
#PBS -o logs/trainjob_full_testq.out
#PBS -e logs/trainjob_full_testq.err

# =============================================================================
# VITS Full Model Training - Amrita HPC (testq - longer walltime)
# =============================================================================
# Queue: testq (max 96h walltime, max 1 GPU per user)
# Resources: 4 CPUs, 1 GPU, 32GB RAM
# Note: testq allows longer jobs but only 1 GPU per user
# =============================================================================

cd $PBS_O_WORKDIR

# Create log directory
mkdir -p logs

echo "============================================================"
echo "  Job Started: $(date)"
echo "  Job ID: $PBS_JOBID"
echo "  Node: $(hostname)"
echo "============================================================"

# Load CUDA module
echo "Loading CUDA module..."
module load cuda11.6/toolkit/11.6.2

# Activate micromamba environment
echo "Activating voicegen environment..."
eval "$(micromamba shell hook --shell bash)"
micromamba activate voicegen

# Print environment info
echo ""
echo "Environment:"
echo "  Python: $(python --version 2>&1)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>&1)"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>&1)"
echo "  GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")' 2>&1)"
echo ""

# Show GPU status
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not available"
echo ""

# Run training with auto-resume support
echo "Starting Full Research Model Training (Stage 9)..."
echo "============================================================"

python train_full.py --config configs/vctk_full.json --model_dir checkpoints/full

echo ""
echo "============================================================"
echo "  Job Finished: $(date)"
echo "============================================================"

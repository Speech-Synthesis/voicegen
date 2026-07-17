#!/bin/bash
#PBS -N vits_training
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb
#PBS -l walltime=72:00:00
#PBS -q gpu
#PBS -o logs/trainjob_hpc.out
#PBS -e logs/trainjob_hpc.err
#PBS -m abe
#PBS -M your.email@university.edu

# =============================================================================
# HPC PBS Training Job Script
# =============================================================================
#
# Features:
#   - Automatic checkpoint resumption after job timeout/restart
#   - Comprehensive logging with system metrics
#   - Progress tracking and ETA estimation
#   - Graceful shutdown on job termination
#
# Submit: qsub trainjob_hpc.sh
# Monitor: qstat -u $USER
# =============================================================================

# Change to working directory
cd $PBS_O_WORKDIR

# Create log directory
mkdir -p logs

# Configuration - MODIFY THESE
CONFIG="configs/vctk_base.json"
MODEL_DIR="checkpoints/vctk_hpc"
TOTAL_STEPS=100000

echo "============================================================"
echo "  VITS Training Job Started"
echo "============================================================"
echo "  Job ID:      $PBS_JOBID"
echo "  Job Name:    $PBS_JOBNAME"
echo "  Node:        $(hostname)"
echo "  Start Time:  $(date)"
echo "  Config:      $CONFIG"
echo "  Model Dir:   $MODEL_DIR"
echo "============================================================"
echo ""

# Load required modules
echo "Loading modules..."
module load cuda11.6/toolkit/11.6.2

# Activate conda/micromamba environment
echo "Activating environment..."
if command -v micromamba &> /dev/null; then
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate voicegen
elif command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate voicegen
else
    echo "Error: No conda/micromamba found"
    exit 1
fi

# Print environment info
echo ""
echo "Environment Info:"
echo "  Python:     $(python --version)"
echo "  PyTorch:    $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA:       $(python -c 'import torch; print(torch.version.cuda)')"
echo "  GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

# Create model directory
mkdir -p "$MODEL_DIR"

# Define trap for graceful shutdown
cleanup() {
    echo ""
    echo "============================================================"
    echo "  Received termination signal - saving checkpoint..."
    echo "============================================================"
    # The Python script handles checkpoint saving on SIGTERM
    sleep 5
    echo "  Job ended at: $(date)"
}
trap cleanup SIGTERM SIGINT

# Run training with auto-resume
echo "Starting training with HPC wrapper..."
echo ""

python hpc_train.py \
    --config "$CONFIG" \
    --model_dir "$MODEL_DIR" \
    --total_steps $TOTAL_STEPS \
    --auto_resume \
    --status_interval 300

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "  Training Job Finished"
echo "============================================================"
echo "  Exit Code:  $EXIT_CODE"
echo "  End Time:   $(date)"
echo "============================================================"

exit $EXIT_CODE

#!/bin/bash
#PBS -N hdvc_full
#PBS -l select=1:ncpus=4:mem=32gb:host=node03
#PBS -l walltime=48:00:00
#PBS -q workq
#PBS -o /home/n_harini/voicegen/logs/trainjob_full.out
#PBS -e /home/n_harini/voicegen/logs/trainjob_full.err

# =============================================================================
# VITS Full Model Training - Amrita HPC
# =============================================================================

cd $PBS_O_WORKDIR

# Create log directory
mkdir -p logs

echo "============================================================"
echo "Job Started: $(date)"
echo "Job ID: $PBS_JOBID"
echo "Node: $(hostname)"
echo "============================================================"

# Load CUDA
echo "Loading CUDA module..."
module load cuda11.6/toolkit/11.6.2

# Activate micromamba environment
echo "Activating voicegen environment..."
eval "$(micromamba shell hook --shell bash)"
micromamba activate voicegen

# Environment Information
echo ""
echo "Environment Information"
echo "-----------------------"

python --version

python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA Version:', torch.version.cuda)"

python - <<'EOF'
import torch
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU: Not Available")
EOF

echo ""

# GPU Status
echo "GPU Status"
echo "----------"
nvidia-smi
echo ""

# Start Training
echo "============================================================"
echo "Starting Full Research Model Training..."
echo "============================================================"

python train_full.py \
    --config configs/vctk_full.json \
    --model_dir checkpoints/full

echo ""
echo "============================================================"
echo "Job Finished: $(date)"
echo "============================================================"

#!/bin/bash -x
#PBS -l ncpus=10
#PBS -l mem=100GB
#PBS -l host=node02
#PBS -q workq

cd $PBS_O_WORKDIR
echo "Job started on $(hostname)"
date

module load cudall.6/toolkit/11.6.2

echo "Checking GPU..."
nvidia-smi

echo "Done."


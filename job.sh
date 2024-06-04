#!/bin/bash
#SBATCH --job-name=MatrixTestNegroTirri
#SBATCH --output=testMatrixOut.txt
#SBATCH --time=02:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1

srun singularity run --bind "$TMPDIR" matrix_multiplication.sif
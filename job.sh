#!/bin/bash
#SBATCH --job-name=MatrixTestNegroTirri
#SBATCH --output=testMatrixOut.txt
#SBATCH --time=00:20
#SBATCH --ntasks=2

export HWLOC_COMPONENTS=-gl
singularity run --bind "$TMPDIR" matrix_multiplication.sif
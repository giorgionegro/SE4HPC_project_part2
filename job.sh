#!/bin/bash
#SBATCH --job-name=MatrixTestNegroTirri
#SBATCH --output=testMatrixOut.txt
#SBATCH --time=00:20
#SBATCH --ntasks=2

module load openmpi
export HWLOC_COMPONENTS=-gl
mpirun -n 2 singularity exec --bind "$TMPDIR" matrix_multiplication.sif /main
#!/bin/bash
# Example Slurm batch script for running the benchmark suite.
# Submit with: sbatch batch_slurm.example.sh

#SBATCH --job-name=ring-bench
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

ulimit -l unlimited

module load nvhpc/24.5
module load cuda/12.4
module load openmpi/4.1.7-nvhpc24.5
module load slurm/17.11.5

export LD_LIBRARY_PATH=/share/compiler/centos7.9/gcc/8.4.0/lib64:$LD_LIBRARY_PATH

export OMPI_MCA_btl=^openib
export OMPI_MCA_pml=ucx
export OMPI_MCA_osc=ucx

export UCX_NET_DEVICES=mlx5_0:1
export UCX_TLS=rc,sm

# Optional: override detected tools if the cluster modules do not expose them in PATH.
# export NVCC=/share/toolkit/cuda/12.4.1/bin/nvcc
# export MPICXX=/share/mpi/openmpi/centos7.9/4.1.7-nvhpc24.5/bin/mpicxx

# Recommended benchmark config for 2 GPUs on 2 nodes.
export NP_LIST="2"
export LAUNCHER="srun"
export SRUN_MPI_TYPE="openmpi"
export BUILD=1
export COLLECT_ENV=1

# Important: start run_suite.sh directly inside sbatch.
# Do not wrap it with "srun ./run_suite.sh", because the benchmark scripts
# already launch MPI steps internally.
./run_suite.sh

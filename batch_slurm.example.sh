#!/bin/bash
# Example Slurm batch script for running the benchmark suite.
# Submit with: sbatch batch_slurm.example.sh

#SBATCH --job-name=ring-bench
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

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

CONFIG=${CONFIG:-benchmark.env}
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/mpi_env.sh"
source_config_preserving_env "${CONFIG}" \
  CONFIG RUN_LABEL RESULTS_ROOT NP_LIST RUN_TYPES WARMUP ITERS BACKENDS \
  COMM_SIZES ATTENTION_SIZES BUILD COLLECT_ENV SRUN SRUN_MPI_TYPE SRUN_EXTRA_ARGS

RUN_LABEL=${RUN_LABEL:-"$(hostname)_$(date +%Y%m%d_%H%M%S)"}
RESULTS_ROOT=${RESULTS_ROOT:-"${SCRIPT_DIR}/results"}
RUN_DIR="${RESULTS_ROOT}/${RUN_LABEL}"
NP_LIST=${NP_LIST:-"2"}
RUN_TYPES=${RUN_TYPES:-"comm attention"}
SRUN=${SRUN:-srun}
SRUN_MPI_TYPE=${SRUN_MPI_TYPE:-openmpi}
SRUN_EXTRA_ARGS=${SRUN_EXTRA_ARGS:-"--ntasks-per-node=1"}

srun_args=()
if [ -n "${SRUN_MPI_TYPE}" ]; then
  srun_args+=(--mpi="${SRUN_MPI_TYPE}")
fi
if [ -n "${SRUN_EXTRA_ARGS}" ]; then
  read -r -a extra_args <<< "${SRUN_EXTRA_ARGS}"
  srun_args+=("${extra_args[@]}")
fi

# Optional: override detected tools if the cluster modules do not expose them in PATH.
# export NVCC=/share/toolkit/cuda/12.4.1/bin/nvcc
# export MPICXX=/share/mpi/openmpi/centos7.9/4.1.7-nvhpc24.5/bin/mpicxx

# Step 1: prepare the run directory, build binaries, and collect env info.
./run_suite.sh

# Step 2: launch MPI tasks from the batch script, not from inside the benchmark scripts.
for np in ${NP_LIST}; do
  for run_type in ${RUN_TYPES}; do
    log="${RUN_DIR}/${run_type}_np${np}.log"
    echo "Running ${run_type} benchmark with NP=${np}..."
    if [ "${#srun_args[@]}" -gt 0 ]; then
      "${SRUN}" "${srun_args[@]}" -n "${np}" \
        env CONFIG="${CONFIG}" NP="${np}" RUN_TYPES="${run_type}" RUN_LABEL="${RUN_LABEL}" RESULTS_ROOT="${RESULTS_ROOT}" \
        ./run_suite.sh > "${log}" 2>&1
    else
      "${SRUN}" -n "${np}" \
        env CONFIG="${CONFIG}" NP="${np}" RUN_TYPES="${run_type}" RUN_LABEL="${RUN_LABEL}" RESULTS_ROOT="${RESULTS_ROOT}" \
        ./run_suite.sh > "${log}" 2>&1
    fi
  done
done

echo "Done. Logs are in ${RUN_DIR}"

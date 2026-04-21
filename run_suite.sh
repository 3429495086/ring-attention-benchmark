#!/bin/bash
# Build, collect environment, and run the benchmark matrix.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/launcher.sh"

CONFIG=${CONFIG:-benchmark.env}
if [ -f "${CONFIG}" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${CONFIG}"
  set +a
fi

ensure_run_suite_not_started_via_parallel_srun

LAUNCHER_KIND="$(resolve_launcher)" || exit 1
build_launcher_command "${LAUNCHER_KIND}" || exit 1
LAUNCHER_SUMMARY="$(launcher_summary "${LAUNCHER_KIND}")"

if [ "${LAUNCHER_KIND}" = "mpirun" ]; then
  MPIRUN="${LAUNCH_BIN}"
fi
if [ "${LAUNCHER_KIND}" = "srun" ]; then
  SRUN="${LAUNCH_BIN}"
fi

RUN_LABEL=${RUN_LABEL:-"$(hostname)_$(date +%Y%m%d_%H%M%S)"}
RESULTS_ROOT=${RESULTS_ROOT:-"${SCRIPT_DIR}/results"}
RUN_DIR="${RESULTS_ROOT}/${RUN_LABEL}"
mkdir -p "${RUN_DIR}"

NP_LIST=${NP_LIST:-"2"}
RUN_TYPES=${RUN_TYPES:-"comm attention"}
WARMUP=${WARMUP:-10}
ITERS=${ITERS:-100}
BACKENDS=${BACKENDS:-"staged staged_Isendrecv cuda_aware cuda_aware_Isendrecv nccl"}
COMM_SIZES=${COMM_SIZES:-"262144 524288 1048576 4194304 16777216"}
ATTENTION_SIZES=${ATTENTION_SIZES:-"262144 524288 1048576 4194304"}
BUILD=${BUILD:-1}
COLLECT_ENV=${COLLECT_ENV:-1}
MPIRUN_EXTRA_ARGS=${MPIRUN_EXTRA_ARGS:-}
HOSTFILE=${HOSTFILE:-}
SRUN_EXTRA_ARGS=${SRUN_EXTRA_ARGS:-}
SRUN_MPI_TYPE=${SRUN_MPI_TYPE:-}

{
  echo "run_label=${RUN_LABEL}"
  echo "date=$(date)"
  echo "hostname=$(hostname)"
  echo "script_dir=${SCRIPT_DIR}"
  echo "launcher=${LAUNCHER_KIND}"
  echo "launcher_request=${LAUNCHER:-auto}"
  echo "launcher_command=${LAUNCHER_SUMMARY}"
  echo "mpirun=${MPIRUN:-}"
  echo "srun=${SRUN:-}"
  echo "hostfile=${HOSTFILE:-}"
  echo "mpirun_extra_args=${MPIRUN_EXTRA_ARGS:-}"
  echo "srun_extra_args=${SRUN_EXTRA_ARGS:-}"
  echo "srun_mpi_type=${SRUN_MPI_TYPE:-}"
  echo "np_list=${NP_LIST}"
  echo "run_types=${RUN_TYPES}"
  echo "warmup=${WARMUP}"
  echo "iters=${ITERS}"
  echo "backends=${BACKENDS}"
  echo "comm_sizes=${COMM_SIZES}"
  echo "attention_sizes=${ATTENTION_SIZES}"
} > "${RUN_DIR}/manifest.txt"

echo "Results directory: ${RUN_DIR}"
echo "Launcher: ${LAUNCHER_SUMMARY}"

if [ "${BUILD}" = "1" ]; then
  echo "Building benchmarks..."
  make all 2>&1 | tee "${RUN_DIR}/build.log"
else
  echo "Skipping build because BUILD=${BUILD}."
fi

if [ "${COLLECT_ENV}" = "1" ]; then
  echo "Collecting local environment..."
  ./collect_env.sh > "${RUN_DIR}/env_local.txt" 2>&1 || true
fi

for np in ${NP_LIST}; do
  for run_type in ${RUN_TYPES}; do
    case "${run_type}" in
      comm)
        log="${RUN_DIR}/comm_np${np}.log"
        echo "Running comm benchmark with NP=${np}..."
        NP="${np}" \
        WARMUP="${WARMUP}" \
        ITERS="${ITERS}" \
        SIZES="${COMM_SIZES}" \
        BACKENDS="${BACKENDS}" \
        LAUNCHER="${LAUNCHER_KIND}" \
        MPIRUN="${MPIRUN}" \
        SRUN="${SRUN:-}" \
        HOSTFILE="${HOSTFILE}" \
        MPIRUN_EXTRA_ARGS="${MPIRUN_EXTRA_ARGS}" \
        SRUN_EXTRA_ARGS="${SRUN_EXTRA_ARGS}" \
        ./benchmark_comm.sh > "${log}" 2>&1
        ;;
      attention)
        log="${RUN_DIR}/attention_np${np}.log"
        echo "Running attention benchmark with NP=${np}..."
        NP="${np}" \
        WARMUP="${WARMUP}" \
        ITERS="${ITERS}" \
        SIZES="${ATTENTION_SIZES}" \
        BACKENDS="${BACKENDS}" \
        LAUNCHER="${LAUNCHER_KIND}" \
        MPIRUN="${MPIRUN}" \
        SRUN="${SRUN:-}" \
        HOSTFILE="${HOSTFILE}" \
        MPIRUN_EXTRA_ARGS="${MPIRUN_EXTRA_ARGS}" \
        SRUN_EXTRA_ARGS="${SRUN_EXTRA_ARGS}" \
        ./benchmark_attention.sh > "${log}" 2>&1
        ;;
      *)
        echo "Skipping unknown RUN_TYPES entry: ${run_type}"
        ;;
    esac
  done
done

echo "Done. Logs are in ${RUN_DIR}"

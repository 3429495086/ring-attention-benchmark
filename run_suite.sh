#!/bin/bash
# Preparation mode: build, collect env, and generate launch examples.
# Worker mode: when started by mpirun/srun, run the selected benchmark family directly.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/mpi_env.sh"

CONFIG=${CONFIG:-benchmark.env}
source_config_preserving_env "${CONFIG}" \
  CONFIG RUN_LABEL RESULTS_ROOT NP_LIST RUN_TYPES WARMUP ITERS BACKENDS \
  COMM_SIZES ATTENTION_SIZES BUILD COLLECT_ENV HOSTFILE MPIRUN MPIRUN_EXTRA_ARGS \
  SRUN SRUN_MPI_TYPE SRUN_EXTRA_ARGS PREPARE_ONLY

RUN_LABEL=${RUN_LABEL:-"$(hostname)_$(date +%Y%m%d_%H%M%S)"}
RESULTS_ROOT=${RESULTS_ROOT:-"${SCRIPT_DIR}/results"}
RUN_DIR="${RESULTS_ROOT}/${RUN_LABEL}"
NP_LIST=${NP_LIST:-"2"}
RUN_TYPES=${RUN_TYPES:-"comm attention"}
WARMUP=${WARMUP:-10}
ITERS=${ITERS:-100}
BACKENDS=${BACKENDS:-"staged staged_Isendrecv cuda_aware cuda_aware_Isendrecv nccl"}
COMM_SIZES=${COMM_SIZES:-"262144 524288 1048576 4194304 16777216"}
ATTENTION_SIZES=${ATTENTION_SIZES:-"262144 524288 1048576 4194304"}
BUILD=${BUILD:-1}
COLLECT_ENV=${COLLECT_ENV:-1}
HOSTFILE=${HOSTFILE:-}
MPIRUN=${MPIRUN:-mpirun}
MPIRUN_EXTRA_ARGS=${MPIRUN_EXTRA_ARGS:-}
SRUN=${SRUN:-srun}
SRUN_MPI_TYPE=${SRUN_MPI_TYPE:-}
SRUN_EXTRA_ARGS=${SRUN_EXTRA_ARGS:-}
PREPARE_ONLY=${PREPARE_ONLY:-0}

if [[ "${CONFIG}" = /* ]]; then
  CONFIG_PATH="${CONFIG}"
else
  CONFIG_PATH="${SCRIPT_DIR}/${CONFIG}"
fi

write_manifest() {
  cat > "${RUN_DIR}/manifest.txt" <<EOF
run_label=${RUN_LABEL}
date=$(date)
hostname=$(hostname)
script_dir=${SCRIPT_DIR}
execution_model=external_launcher
config=${CONFIG}
results_root=${RESULTS_ROOT}
run_dir=${RUN_DIR}
np_list=${NP_LIST}
run_types=${RUN_TYPES}
warmup=${WARMUP}
iters=${ITERS}
backends=${BACKENDS}
comm_sizes=${COMM_SIZES}
attention_sizes=${ATTENTION_SIZES}
build=${BUILD}
collect_env=${COLLECT_ENV}
mpirun=${MPIRUN}
hostfile=${HOSTFILE}
mpirun_extra_args=${MPIRUN_EXTRA_ARGS}
srun=${SRUN}
srun_mpi_type=${SRUN_MPI_TYPE}
srun_extra_args=${SRUN_EXTRA_ARGS}
EOF
}

write_run_context() {
  cat > "${RUN_DIR}/run_context.env" <<EOF
RUN_LABEL="${RUN_LABEL}"
RESULTS_ROOT="${RESULTS_ROOT}"
RUN_DIR="${RUN_DIR}"
CONFIG="${CONFIG}"
NP_LIST="${NP_LIST}"
RUN_TYPES="${RUN_TYPES}"
EOF
}

write_launch_examples() {
  local mpirun_prefix="${MPIRUN}"
  local srun_prefix="${SRUN}"

  if [ -n "${HOSTFILE}" ]; then
    mpirun_prefix="${mpirun_prefix} --hostfile ${HOSTFILE}"
  fi
  if [ -n "${MPIRUN_EXTRA_ARGS}" ]; then
    mpirun_prefix="${mpirun_prefix} ${MPIRUN_EXTRA_ARGS}"
  fi
  if [ -n "${SRUN_MPI_TYPE}" ]; then
    srun_prefix="${srun_prefix} --mpi=${SRUN_MPI_TYPE}"
  fi
  if [ -n "${SRUN_EXTRA_ARGS}" ]; then
    srun_prefix="${srun_prefix} ${SRUN_EXTRA_ARGS}"
  fi

  {
    echo "# Ring Attention benchmark launch examples"
    echo "# Preparation already created: ${RUN_DIR}"
    echo "# The MPI/Slurm launcher is outside the benchmark scripts."
    echo ""
    echo "# OpenMPI / mpirun examples"
    for np in ${NP_LIST}; do
      for run_type in ${RUN_TYPES}; do
        echo "${mpirun_prefix} -np ${np} env CONFIG=\"${CONFIG_PATH}\" NP=\"${np}\" RUN_TYPES=\"${run_type}\" RUN_LABEL=\"${RUN_LABEL}\" RESULTS_ROOT=\"${RESULTS_ROOT}\" \"${SCRIPT_DIR}/run_suite.sh\" > \"${RUN_DIR}/${run_type}_np${np}.log\" 2>&1"
      done
    done
    echo ""
    echo "# Slurm / srun examples"
    for np in ${NP_LIST}; do
      for run_type in ${RUN_TYPES}; do
        echo "${srun_prefix} -n ${np} env CONFIG=\"${CONFIG_PATH}\" NP=\"${np}\" RUN_TYPES=\"${run_type}\" RUN_LABEL=\"${RUN_LABEL}\" RESULTS_ROOT=\"${RESULTS_ROOT}\" \"${SCRIPT_DIR}/run_suite.sh\" > \"${RUN_DIR}/${run_type}_np${np}.log\" 2>&1"
      done
    done
  } > "${RUN_DIR}/launch_examples.txt"
}

prepare_suite() {
  mkdir -p "${RUN_DIR}"
  write_manifest
  write_run_context
  write_launch_examples

  echo "Results directory: ${RUN_DIR}"

  if [ "${BUILD}" = "1" ]; then
    echo "Building benchmarks..."
    make CONFIG="${CONFIG}" all 2>&1 | tee "${RUN_DIR}/build.log"
  else
    echo "Skipping build because BUILD=${BUILD}."
  fi

  if [ "${COLLECT_ENV}" = "1" ]; then
    echo "Collecting local environment..."
    ./collect_env.sh > "${RUN_DIR}/env_local.txt" 2>&1 || true
  fi

  echo "Preparation complete."
  echo "Launch the MPI jobs externally. Example commands are in ${RUN_DIR}/launch_examples.txt"
}

run_worker() {
  local detected_np=""
  detected_np="$(detect_world_size 2>/dev/null || true)"

  if [ -z "${detected_np}" ]; then
    echo "ERROR: run_suite.sh worker mode requires mpirun/srun to launch the tasks externally." >&2
    exit 1
  fi

  if is_primary_rank; then
    echo "========== Ring Attention Suite =========="
    echo "Date: $(date)"
    echo "Hostname: $(hostname)"
    echo "np=${detected_np} run_types=${RUN_TYPES}"
    echo "results_dir=${RUN_DIR}"
    echo ""
  fi

  for run_type in ${RUN_TYPES}; do
    case "${run_type}" in
      comm)
        WARMUP="${WARMUP}" \
        ITERS="${ITERS}" \
        BACKENDS="${BACKENDS}" \
        SIZES="${COMM_SIZES}" \
        NP="${detected_np}" \
        CONFIG="${CONFIG}" \
        ./benchmark_comm.sh
        ;;
      attention)
        WARMUP="${WARMUP}" \
        ITERS="${ITERS}" \
        BACKENDS="${BACKENDS}" \
        SIZES="${ATTENTION_SIZES}" \
        NP="${detected_np}" \
        CONFIG="${CONFIG}" \
        ./benchmark_attention.sh
        ;;
      *)
        echo "ERROR: unknown RUN_TYPES entry: ${run_type}" >&2
        exit 1
        ;;
    esac
  done

  if is_primary_rank; then
    echo "========== Suite Done =========="
    echo "Finished at: $(date)"
  fi
}

if [ "${PREPARE_ONLY}" = "1" ] || ! under_mpi_launcher; then
  prepare_suite
else
  run_worker
fi

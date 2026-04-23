#!/bin/bash
# Preparation-only mode: build, collect env, and generate launch examples.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/mpi_env.sh"

CONFIG=${CONFIG:-benchmark.env}
source_config_preserving_env "${CONFIG}" \
  CONFIG RUN_LABEL RESULTS_ROOT NP_LIST RUN_TYPES WARMUP ITERS BACKENDS ATTENTION_BACKENDS \
  COMM_SIZES ATTENTION_SIZES BUILD COLLECT_ENV HOSTFILE MPIRUN MPIRUN_EXTRA_ARGS \
  SRUN SRUN_MPI_TYPE SRUN_EXTRA_ARGS PREPARE_ONLY
configure_runtime_env

RUN_LABEL=${RUN_LABEL:-"$(hostname)_$(date +%Y%m%d_%H%M%S)"}
RESULTS_ROOT=${RESULTS_ROOT:-"${SCRIPT_DIR}/results"}
RUN_DIR="${RESULTS_ROOT}/${RUN_LABEL}"
NP_LIST=${NP_LIST:-"2"}
RUN_TYPES=${RUN_TYPES:-"comm attention"}
WARMUP=${WARMUP:-10}
ITERS=${ITERS:-100}
BACKENDS=${BACKENDS:-"staged staged_Isendrecv cuda_aware cuda_aware_Isendrecv nccl"}
ATTENTION_BACKENDS=${ATTENTION_BACKENDS:-"staged staged_Isendrecv staged_Isendrecv_overlap cuda_aware cuda_aware_Isendrecv cuda_aware_Isendrecv_overlap nccl"}
COMM_SIZES=${COMM_SIZES:-"262144 524288 1048576 4194304 16777216"}
ATTENTION_SIZES=${ATTENTION_SIZES:-"262144 524288 1048576"}
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
attention_backends=${ATTENTION_BACKENDS}
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
BACKENDS="${BACKENDS}"
ATTENTION_BACKENDS="${ATTENTION_BACKENDS}"
EOF
}

write_launch_examples() {
  local mpirun_prefix=""
  local srun_prefix=""

  mpirun_prefix="env CONFIG=\"${CONFIG_PATH}\" RUN_LABEL=\"${RUN_LABEL}\" RESULTS_ROOT=\"${RESULTS_ROOT}\" LAUNCHER=\"mpirun\" MPIRUN=\"${MPIRUN}\""
  if [ -n "${HOSTFILE}" ]; then
    mpirun_prefix="${mpirun_prefix} HOSTFILE=\"${HOSTFILE}\""
  fi
  if [ -n "${MPIRUN_EXTRA_ARGS}" ]; then
    mpirun_prefix="${mpirun_prefix} MPIRUN_EXTRA_ARGS=\"${MPIRUN_EXTRA_ARGS}\""
  fi

  srun_prefix="env CONFIG=\"${CONFIG_PATH}\" RUN_LABEL=\"${RUN_LABEL}\" RESULTS_ROOT=\"${RESULTS_ROOT}\" LAUNCHER=\"srun\" SRUN=\"${SRUN}\""
  if [ -n "${SRUN_MPI_TYPE}" ]; then
    srun_prefix="${srun_prefix} SRUN_MPI_TYPE=\"${SRUN_MPI_TYPE}\""
  fi
  if [ -n "${SRUN_EXTRA_ARGS}" ]; then
    srun_prefix="${srun_prefix} SRUN_EXTRA_ARGS=\"${SRUN_EXTRA_ARGS}\""
  fi

  {
    echo "# Ring Attention benchmark launch examples"
    echo "# Preparation already created: ${RUN_DIR}"
    echo "# The launcher should call the benchmark wrappers directly."
    echo ""
    echo "# OpenMPI / mpirun examples"
    for np in ${NP_LIST}; do
      for run_type in ${RUN_TYPES}; do
        case "${run_type}" in
          comm)
            echo "${mpirun_prefix} NP=\"${np}\" \"${SCRIPT_DIR}/benchmark_comm.sh\" > \"${RUN_DIR}/${run_type}_np${np}.log\" 2>&1"
            ;;
          attention)
            echo "${mpirun_prefix} NP=\"${np}\" \"${SCRIPT_DIR}/benchmark_attention.sh\" > \"${RUN_DIR}/${run_type}_np${np}.log\" 2>&1"
            ;;
        esac
      done
    done
    echo ""
    echo "# Slurm / srun examples"
    for np in ${NP_LIST}; do
      for run_type in ${RUN_TYPES}; do
        case "${run_type}" in
          comm)
            echo "${srun_prefix} NP=\"${np}\" \"${SCRIPT_DIR}/benchmark_comm.sh\" > \"${RUN_DIR}/${run_type}_np${np}.log\" 2>&1"
            ;;
          attention)
            echo "${srun_prefix} NP=\"${np}\" \"${SCRIPT_DIR}/benchmark_attention.sh\" > \"${RUN_DIR}/${run_type}_np${np}.log\" 2>&1"
            ;;
        esac
      done
    done
  } > "${RUN_DIR}/launch_examples.txt"
}

build_target_for_backend() {
  local run_type="$1"
  local backend="$2"

  case "${run_type}:${backend}" in
    comm:staged) printf '%s\n' "bin/ring_loop_staged" ;;
    comm:staged_Isendrecv) printf '%s\n' "bin/ring_loop_staged_Isendrecv" ;;
    comm:cuda_aware) printf '%s\n' "bin/ring_loop_cuda_aware" ;;
    comm:cuda_aware_Isendrecv) printf '%s\n' "bin/ring_loop_cuda_aware_Isendrecv" ;;
    comm:nccl) printf '%s\n' "bin/ring_loop_nccl" ;;
    attention:staged) printf '%s\n' "bin/ring_attention_staged_gpu_bench" ;;
    attention:staged_Isendrecv) printf '%s\n' "bin/ring_attention_staged_Isendrecv_gpu_bench" ;;
    attention:staged_Isendrecv_overlap) printf '%s\n' "bin/ring_attention_staged_Isendrecv_overlap_gpu_bench" ;;
    attention:cuda_aware) printf '%s\n' "bin/ring_attention_cuda_aware_gpu_bench" ;;
    attention:cuda_aware_Isendrecv) printf '%s\n' "bin/ring_attention_cuda_aware_Isendrecv_gpu_bench" ;;
    attention:cuda_aware_Isendrecv_overlap) printf '%s\n' "bin/ring_attention_cuda_aware_Isendrecv_overlap_gpu_bench" ;;
    attention:nccl) printf '%s\n' "bin/ring_attention_nccl_gpu_bench" ;;
    *) return 1 ;;
  esac
}

build_requested_targets() {
  local -a targets=()
  local run_type=""
  local backend=""
  local target=""

  for run_type in ${RUN_TYPES}; do
    case "${run_type}" in
      comm)
        for backend in ${BACKENDS}; do
          if target="$(build_target_for_backend "${run_type}" "${backend}")"; then
            targets+=("${target}")
          else
            echo "WARN: skipping unknown comm backend during build: ${backend}"
          fi
        done
        ;;
      attention)
        for backend in ${ATTENTION_BACKENDS}; do
          if target="$(build_target_for_backend "${run_type}" "${backend}")"; then
            targets+=("${target}")
          else
            echo "WARN: skipping unknown attention backend during build: ${backend}"
          fi
        done
        ;;
      *)
        echo "WARN: skipping unknown RUN_TYPES entry during build: ${run_type}"
        ;;
    esac
  done

  if [ "${#targets[@]}" -eq 0 ]; then
    echo "No build targets selected from RUN_TYPES/BACKENDS."
    return 0
  fi

  mapfile -t targets < <(printf '%s\n' "${targets[@]}" | awk '!seen[$0]++')
  make CONFIG="${CONFIG}" "${targets[@]}"
}

prepare_suite() {
  mkdir -p "${RUN_DIR}"
  write_manifest
  write_run_context
  write_launch_examples

  echo "Results directory: ${RUN_DIR}"

  if [ "${BUILD}" = "1" ]; then
    echo "Building benchmarks..."
    build_requested_targets 2>&1 | tee "${RUN_DIR}/build.log"
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

if under_mpi_launcher; then
  echo "ERROR: run_suite.sh is preparation-only." >&2
  echo "Use benchmark_comm.sh / benchmark_attention.sh from a normal shell, or the example launcher scripts." >&2
  exit 1
fi

prepare_suite

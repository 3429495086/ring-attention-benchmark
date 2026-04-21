#!/bin/bash
# benchmark_comm.sh - Launch communication-only (loop) benchmarks from a normal shell.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/mpi_env.sh"

CONFIG=${CONFIG:-benchmark.env}
source_config_preserving_env "${CONFIG}" CONFIG NP WARMUP ITERS SIZES BACKENDS \
  LAUNCHER MPIRUN HOSTFILE MPIRUN_EXTRA_ARGS SRUN SRUN_MPI_TYPE SRUN_EXTRA_ARGS
configure_runtime_env

if under_mpi_launcher; then
  echo "ERROR: benchmark_comm.sh should be run from a normal shell; it launches mpirun/srun itself." >&2
  exit 1
fi

NP=${NP:-2}
WARMUP=${WARMUP:-10}
ITERS=${ITERS:-100}
SIZES=${SIZES:-"262144 524288 1048576 4194304 16777216"}
BACKENDS=${BACKENDS:-"staged staged_Isendrecv cuda_aware cuda_aware_Isendrecv nccl"}
LAUNCHER=${LAUNCHER:-mpirun}
MPIRUN=${MPIRUN:-mpirun}
HOSTFILE=${HOSTFILE:-}
MPIRUN_EXTRA_ARGS=${MPIRUN_EXTRA_ARGS:-}
SRUN=${SRUN:-srun}
SRUN_MPI_TYPE=${SRUN_MPI_TYPE:-}
SRUN_EXTRA_ARGS=${SRUN_EXTRA_ARGS:-}

BIN_DIR="${SCRIPT_DIR}/bin"

mpirun_args=()
if [ -n "${HOSTFILE}" ]; then
  mpirun_args+=(--hostfile "${HOSTFILE}")
fi
if [ -n "${MPIRUN_EXTRA_ARGS}" ]; then
  read -r -a extra_args <<< "${MPIRUN_EXTRA_ARGS}"
  mpirun_args+=("${extra_args[@]}")
fi

srun_args=()
if [ -n "${SRUN_MPI_TYPE}" ]; then
  srun_args+=(--mpi="${SRUN_MPI_TYPE}")
fi
if [ -n "${SRUN_EXTRA_ARGS}" ]; then
  read -r -a extra_args <<< "${SRUN_EXTRA_ARGS}"
  srun_args+=("${extra_args[@]}")
fi

backend_exe() {
  case "$1" in
    staged) printf '%s\n' "${BIN_DIR}/ring_loop_staged" ;;
    staged_Isendrecv) printf '%s\n' "${BIN_DIR}/ring_loop_staged_Isendrecv" ;;
    cuda_aware) printf '%s\n' "${BIN_DIR}/ring_loop_cuda_aware" ;;
    cuda_aware_Isendrecv) printf '%s\n' "${BIN_DIR}/ring_loop_cuda_aware_Isendrecv" ;;
    nccl) printf '%s\n' "${BIN_DIR}/ring_loop_nccl" ;;
    *) return 1 ;;
  esac
}

run_backend() {
  local exe="$1"
  local size="$2"

  case "${LAUNCHER}" in
    mpirun)
      if [ "${#mpirun_args[@]}" -gt 0 ]; then
        "${MPIRUN}" "${mpirun_args[@]}" -np "${NP}" "${exe}" "${size}" "${WARMUP}" "${ITERS}"
      else
        "${MPIRUN}" -np "${NP}" "${exe}" "${size}" "${WARMUP}" "${ITERS}"
      fi
      ;;
    srun)
      if [ "${#srun_args[@]}" -gt 0 ]; then
        "${SRUN}" "${srun_args[@]}" -n "${NP}" "${exe}" "${size}" "${WARMUP}" "${ITERS}"
      else
        "${SRUN}" -n "${NP}" "${exe}" "${size}" "${WARMUP}" "${ITERS}"
      fi
      ;;
    none)
      if [ "${NP}" != "1" ]; then
        echo "ERROR: LAUNCHER=none requires NP=1." >&2
        return 1
      fi
      "${exe}" "${size}" "${WARMUP}" "${ITERS}"
      ;;
    *)
      echo "ERROR: unknown LAUNCHER=${LAUNCHER}. Use mpirun, srun, or none." >&2
      return 1
      ;;
  esac
}

echo "========== Communication (Loop) Benchmark =========="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "launcher=${LAUNCHER} np=${NP} warmup=${WARMUP} iters=${ITERS}"
echo "mpirun=${MPIRUN}"
echo "sizes=${SIZES}"
echo "backends=${BACKENDS}"
echo ""

for backend in ${BACKENDS}; do
  if ! exe="$(backend_exe "${backend}")"; then
    echo "SKIP backend=${backend} (unknown backend)"
    echo ""
    continue
  fi

  if [[ ! -x "${exe}" ]]; then
    echo "SKIP backend=${backend} (executable not found: ${exe})"
    echo ""
    continue
  fi

  echo "===== backend=${backend} ====="
  for size in ${SIZES}; do
    echo "--- kv_size=${size} bytes ---"
    if ! run_backend "${exe}" "${size}"; then
      echo "ERROR: ${backend} failed with kv_size=${size}"
      exit 1
    fi
  done
  echo ""
done

echo "========== Done =========="
echo "Finished at: $(date)"

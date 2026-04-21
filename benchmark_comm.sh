#!/bin/bash
# benchmark_comm.sh - Run communication-only (loop) benchmarks inside an externally launched MPI job.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/mpi_env.sh"

CONFIG=${CONFIG:-benchmark.env}
source_config_preserving_env "${CONFIG}" CONFIG NP WARMUP ITERS SIZES BACKENDS

DETECTED_NP="$(detect_world_size 2>/dev/null || true)"
NP=${NP:-${DETECTED_NP:-1}}
if [ -n "${DETECTED_NP}" ] && [ "${NP}" != "${DETECTED_NP}" ]; then
  log_primary "WARN: NP=${NP} differs from detected world size=${DETECTED_NP}; using detected size."
  NP="${DETECTED_NP}"
fi

if [ "${NP}" -gt 1 ] && ! under_mpi_launcher; then
  echo "ERROR: benchmark_comm.sh must be launched externally with mpirun/srun when NP=${NP}." >&2
  exit 1
fi

WARMUP=${WARMUP:-10}
ITERS=${ITERS:-100}
SIZES=${SIZES:-"262144 524288 1048576 4194304 16777216"}
BACKENDS=${BACKENDS:-"staged staged_Isendrecv cuda_aware cuda_aware_Isendrecv nccl"}

BIN_DIR="${SCRIPT_DIR}/bin"

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

log_primary "========== Communication (Loop) Benchmark =========="
log_primary "Date: $(date)"
log_primary "Hostname: $(hostname)"
log_primary "np=${NP} warmup=${WARMUP} iters=${ITERS}"
log_primary "sizes=${SIZES}"
log_primary "backends=${BACKENDS}"
log_primary ""

for backend in ${BACKENDS}; do
  if ! exe="$(backend_exe "${backend}")"; then
    log_primary "SKIP backend=${backend} (unknown backend)"
    log_primary ""
    continue
  fi

  if [[ ! -x "${exe}" ]]; then
    log_primary "SKIP backend=${backend} (executable not found: ${exe})"
    log_primary ""
    continue
  fi

  log_primary "===== backend=${backend} ====="
  for size in ${SIZES}; do
    log_primary "--- kv_size=${size} bytes ---"
    if ! "${exe}" "${size}" "${WARMUP}" "${ITERS}"; then
      log_primary "ERROR: ${backend} failed with kv_size=${size}"
      exit 1
    fi
  done
  log_primary ""
done

log_primary "========== Done =========="
log_primary "Finished at: $(date)"

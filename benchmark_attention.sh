#!/bin/bash
# benchmark_attention.sh - Run Ring Attention benchmarks across all backends
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/launcher.sh"

ensure_benchmark_wrapper_not_started_via_parallel_srun "benchmark_attention.sh"

# ============ Configuration ============
NP=${NP:-2}
WARMUP=${WARMUP:-10}
ITERS=${ITERS:-100}
MPIRUN_EXTRA_ARGS=${MPIRUN_EXTRA_ARGS:-}
HOSTFILE=${HOSTFILE:-}
SRUN_EXTRA_ARGS=${SRUN_EXTRA_ARGS:-}
LAUNCHER_KIND="$(resolve_launcher)" || exit 1
build_launcher_command "${LAUNCHER_KIND}" || exit 1

# KV shard sizes in bytes (must be multiple of 256 = 64 * sizeof(float))
SIZES=${SIZES:-"262144 524288 1048576 4194304"}
BACKENDS=${BACKENDS:-"staged staged_Isendrecv cuda_aware cuda_aware_Isendrecv nccl"}

BIN_DIR="${SCRIPT_DIR}/bin"

# ============ Backend executables ============
declare -A EXES=(
  [staged]=${BIN_DIR}/ring_attention_staged_gpu_bench
  [staged_Isendrecv]=${BIN_DIR}/ring_attention_staged_Isendrecv_gpu_bench
  [cuda_aware]=${BIN_DIR}/ring_attention_cuda_aware_gpu_bench
  [cuda_aware_Isendrecv]=${BIN_DIR}/ring_attention_cuda_aware_Isendrecv_gpu_bench
  [nccl]=${BIN_DIR}/ring_attention_nccl_gpu_bench
)

# ============ Run benchmarks ============
echo "========== Ring Attention Benchmark =========="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "launcher: ${LAUNCHER_KIND}"
echo "launcher_bin: ${LAUNCH_BIN}"
echo "launcher_args: ${LAUNCH_ARGS[*]:-(none)}"
echo "np=${NP} warmup=${WARMUP} iters=${ITERS}"
echo "sizes=${SIZES}"
echo "backends=${BACKENDS}"
echo ""

for backend in ${BACKENDS}; do
  if [[ -z "${EXES[$backend]+set}" ]]; then
    echo "SKIP backend=${backend} (unknown backend)"
    echo ""
    continue
  fi

  exe="${EXES[$backend]}"
  if [[ ! -x "${exe}" ]]; then
    echo "SKIP backend=${backend} (executable not found: ${exe})"
    echo ""
    continue
  fi

  echo "===== backend=${backend} ====="
  for size in ${SIZES}; do
    echo "--- kv_size=${size} bytes ---"
    case "${LAUNCHER_KIND}" in
      mpirun)
        "${LAUNCH_BIN}" "${LAUNCH_ARGS[@]}" -np "${NP}" "${exe}" "${size}" "${WARMUP}" "${ITERS}" || {
            echo "ERROR: ${backend} failed with kv_size=${size}"
        }
        ;;
      srun)
        "${LAUNCH_BIN}" "${LAUNCH_ARGS[@]}" -n "${NP}" "${exe}" "${size}" "${WARMUP}" "${ITERS}" || {
            echo "ERROR: ${backend} failed with kv_size=${size}"
        }
        ;;
      *)
        echo "ERROR: unsupported launcher=${LAUNCHER_KIND}"
        exit 1
        ;;
    esac
  done
  echo ""
done

echo "========== Done =========="
echo "Finished at: $(date)"

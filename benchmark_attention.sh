#!/bin/bash
# benchmark_attention.sh - Run Ring Attention benchmarks across all backends
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# ============ Configuration ============
MPIRUN="${MPIRUN:-}"
if [ -z "$MPIRUN" ]; then
    for candidate in \
        /nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5/bin/mpirun \
        /usr/bin/mpirun \
        mpirun; do
        if command -v "$candidate" &> /dev/null; then
            MPIRUN="$candidate"
            break
        fi
    done
fi

if [ -z "$MPIRUN" ]; then
    echo "ERROR: mpirun not found. Set MPIRUN environment variable."
    exit 1
fi

NP=${NP:-2}
WARMUP=${WARMUP:-10}
ITERS=${ITERS:-100}
MPIRUN_EXTRA_ARGS=${MPIRUN_EXTRA_ARGS:-}
HOSTFILE=${HOSTFILE:-}

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

MPIRUN_ARGS=()
if [ -n "${HOSTFILE}" ]; then
  MPIRUN_ARGS+=(--hostfile "${HOSTFILE}")
fi
if [ -n "${MPIRUN_EXTRA_ARGS}" ]; then
  # Simple whitespace splitting is intended for scheduler-style flags.
  read -r -a EXTRA_ARGS <<< "${MPIRUN_EXTRA_ARGS}"
  MPIRUN_ARGS+=("${EXTRA_ARGS[@]}")
fi

# ============ Run benchmarks ============
echo "========== Ring Attention Benchmark =========="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "mpirun: ${MPIRUN}"
echo "mpirun_args: ${MPIRUN_ARGS[*]:-(none)}"
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
    "${MPIRUN}" "${MPIRUN_ARGS[@]}" -np "${NP}" "${exe}" "${size}" "${WARMUP}" "${ITERS}" || {
        echo "ERROR: ${backend} failed with kv_size=${size}"
    }
  done
  echo ""
done

echo "========== Done =========="
echo "Finished at: $(date)"

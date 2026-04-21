#!/bin/bash
# Example OpenMPI launcher for the benchmark matrix.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

CONFIG=${CONFIG:-benchmark.env}
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/mpi_env.sh"
source_config_preserving_env "${CONFIG}" \
  CONFIG RUN_LABEL RESULTS_ROOT NP_LIST RUN_TYPES WARMUP ITERS BACKENDS \
  COMM_SIZES ATTENTION_SIZES BUILD COLLECT_ENV MPIRUN HOSTFILE MPIRUN_EXTRA_ARGS

RUN_LABEL=${RUN_LABEL:-"$(hostname)_$(date +%Y%m%d_%H%M%S)"}
RESULTS_ROOT=${RESULTS_ROOT:-"${SCRIPT_DIR}/results"}
RUN_DIR="${RESULTS_ROOT}/${RUN_LABEL}"
NP_LIST=${NP_LIST:-"2"}
RUN_TYPES=${RUN_TYPES:-"comm attention"}
MPIRUN=${MPIRUN:-mpirun}
HOSTFILE=${HOSTFILE:-}
MPIRUN_EXTRA_ARGS=${MPIRUN_EXTRA_ARGS:-}

mpirun_args=()
if [ -n "${HOSTFILE}" ]; then
  mpirun_args+=(--hostfile "${HOSTFILE}")
fi
if [ -n "${MPIRUN_EXTRA_ARGS}" ]; then
  read -r -a extra_args <<< "${MPIRUN_EXTRA_ARGS}"
  mpirun_args+=("${extra_args[@]}")
fi

./run_suite.sh

for np in ${NP_LIST}; do
  for run_type in ${RUN_TYPES}; do
    log="${RUN_DIR}/${run_type}_np${np}.log"
    echo "Running ${run_type} benchmark with NP=${np}..."
    if [ "${#mpirun_args[@]}" -gt 0 ]; then
      "${MPIRUN}" "${mpirun_args[@]}" -np "${np}" \
        env CONFIG="${CONFIG}" NP="${np}" RUN_TYPES="${run_type}" RUN_LABEL="${RUN_LABEL}" RESULTS_ROOT="${RESULTS_ROOT}" \
        ./run_suite.sh > "${log}" 2>&1
    else
      "${MPIRUN}" -np "${np}" \
        env CONFIG="${CONFIG}" NP="${np}" RUN_TYPES="${run_type}" RUN_LABEL="${RUN_LABEL}" RESULTS_ROOT="${RESULTS_ROOT}" \
        ./run_suite.sh > "${log}" 2>&1
    fi
  done
done

echo "Done. Logs are in ${RUN_DIR}"

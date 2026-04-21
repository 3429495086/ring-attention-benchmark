#!/bin/bash

find_mpirun() {
  if [ -n "${MPIRUN:-}" ]; then
    echo "${MPIRUN}"
    return 0
  fi

  for candidate in \
      /nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5/bin/mpirun \
      /usr/bin/mpirun \
      mpirun; do
    if command -v "${candidate}" &> /dev/null; then
      echo "${candidate}"
      return 0
    fi
  done

  return 1
}

find_srun() {
  if [ -n "${SRUN:-}" ]; then
    echo "${SRUN}"
    return 0
  fi

  if command -v srun &> /dev/null; then
    command -v srun
    return 0
  fi

  return 1
}

detect_default_srun_mpi_type() {
  if [ -n "${SRUN_MPI_TYPE:-}" ]; then
    echo "${SRUN_MPI_TYPE}"
    return 0
  fi

  local mpirun_bin=""
  mpirun_bin="$(find_mpirun 2>/dev/null || true)"
  if [ -n "${mpirun_bin}" ] && "${mpirun_bin}" --version 2>/dev/null | head -1 | grep -qi "open mpi"; then
    echo "openmpi"
    return 0
  fi

  return 1
}

resolve_launcher() {
  local mode="${LAUNCHER:-auto}"

  case "${mode}" in
    auto)
      if [ -n "${SLURM_JOB_ID:-}" ]; then
        if find_srun &> /dev/null; then
          echo "srun"
          return 0
        fi
      fi
      if find_mpirun &> /dev/null; then
        echo "mpirun"
        return 0
      fi
      ;;
    mpirun)
      if find_mpirun &> /dev/null; then
        echo "mpirun"
        return 0
      fi
      ;;
    srun)
      if find_srun &> /dev/null; then
        echo "srun"
        return 0
      fi
      ;;
    *)
      echo "ERROR: unknown LAUNCHER=${mode}. Use auto, mpirun, or srun." >&2
      return 1
      ;;
  esac

  echo "ERROR: unable to resolve launcher for LAUNCHER=${mode}." >&2
  return 1
}

ensure_run_suite_not_started_via_parallel_srun() {
  if [ -n "${SLURM_PROCID:-}" ] && [ "${RUN_SUITE_ALLOW_OUTER_SRUN:-0}" != "1" ]; then
    echo "ERROR: run_suite.sh should be started once inside an sbatch allocation, not via 'srun ./run_suite.sh'." >&2
    echo "Use sbatch to allocate resources, then call './run_suite.sh' from the batch script." >&2
    echo "Current SLURM_PROCID=${SLURM_PROCID} suggests an outer srun step is already active." >&2
    exit 1
  fi
}

ensure_benchmark_wrapper_not_started_via_parallel_srun() {
  local script_name="$1"

  if [ -n "${SLURM_PROCID:-}" ] && [ "${RUN_SUITE_ALLOW_OUTER_SRUN:-0}" != "1" ]; then
    echo "ERROR: ${script_name} should be started from the batch shell, not via 'srun ${script_name}'." >&2
    echo "Use './run_suite.sh' inside sbatch, or launch the benchmark binary directly with srun." >&2
    echo "Current SLURM_PROCID=${SLURM_PROCID} suggests an outer srun step is already active." >&2
    exit 1
  fi
}

build_launcher_command() {
  local launcher="$1"

  LAUNCH_BIN=""
  LAUNCH_ARGS=()

  case "${launcher}" in
    mpirun)
      LAUNCH_BIN="$(find_mpirun)" || {
        echo "ERROR: mpirun not found." >&2
        return 1
      }

      if [ -n "${HOSTFILE:-}" ]; then
        LAUNCH_ARGS+=(--hostfile "${HOSTFILE}")
      fi
      if [ -n "${MPIRUN_EXTRA_ARGS:-}" ]; then
        read -r -a extra_args <<< "${MPIRUN_EXTRA_ARGS}"
        LAUNCH_ARGS+=("${extra_args[@]}")
      fi
      ;;
    srun)
      LAUNCH_BIN="$(find_srun)" || {
        echo "ERROR: srun not found." >&2
        return 1
      }

      local mpi_type=""
      mpi_type="$(detect_default_srun_mpi_type 2>/dev/null || true)"
      if [ -n "${mpi_type}" ]; then
        LAUNCH_ARGS+=(--mpi="${mpi_type}")
      fi
      if [ -n "${SRUN_EXTRA_ARGS:-}" ]; then
        read -r -a extra_args <<< "${SRUN_EXTRA_ARGS}"
        LAUNCH_ARGS+=("${extra_args[@]}")
      fi
      ;;
    *)
      echo "ERROR: build_launcher_command got unsupported launcher=${launcher}." >&2
      return 1
      ;;
  esac
}

launcher_summary() {
  local launcher="$1"
  local joined_args="${LAUNCH_ARGS[*]:-(none)}"
  printf '%s\n' "${launcher} ${joined_args}"
}

#!/bin/bash

path_contains() {
  local needle="$1"
  local value="$2"
  local entry=""
  local -a entries=()

  if [ -z "${value}" ]; then
    return 1
  fi

  IFS=':' read -r -a entries <<< "${value}"
  for entry in "${entries[@]}"; do
    if [ "${entry}" = "${needle}" ]; then
      return 0
    fi
  done

  return 1
}

prepend_path_var() {
  local var_name="$1"
  local dir="$2"
  local current_value="${!var_name:-}"

  if [ -z "${dir}" ] || [ ! -d "${dir}" ]; then
    return 0
  fi

  if path_contains "${dir}" "${current_value}"; then
    return 0
  fi

  if [ -n "${current_value}" ]; then
    printf -v "${var_name}" '%s:%s' "${dir}" "${current_value}"
  else
    printf -v "${var_name}" '%s' "${dir}"
  fi
  export "${var_name}"
}

configure_runtime_env() {
  if [ -n "${MPI_HOME:-}" ]; then
    prepend_path_var PATH "${MPI_HOME}/bin"
    prepend_path_var LD_LIBRARY_PATH "${MPI_HOME}/lib64"
    prepend_path_var LD_LIBRARY_PATH "${MPI_HOME}/lib"

    if [ -z "${MPIRUN:-}" ] && [ -x "${MPI_HOME}/bin/mpirun" ]; then
      MPIRUN="${MPI_HOME}/bin/mpirun"
      export MPIRUN
    fi

    if [ -z "${OPAL_PREFIX:-}" ]; then
      OPAL_PREFIX="${MPI_HOME}"
      export OPAL_PREFIX
    fi
  fi

  if [ -n "${NCCL_HOME:-}" ]; then
    prepend_path_var LD_LIBRARY_PATH "${NCCL_HOME}/lib64"
    prepend_path_var LD_LIBRARY_PATH "${NCCL_HOME}/lib"
  fi

  if [ -n "${CUDA_HOME:-}" ]; then
    prepend_path_var PATH "${CUDA_HOME}/bin"
    prepend_path_var LD_LIBRARY_PATH "${CUDA_HOME}/lib64"
    prepend_path_var LD_LIBRARY_PATH "${CUDA_HOME}/lib"
  fi
}

detect_world_rank() {
  for value in \
    "${OMPI_COMM_WORLD_RANK:-}" \
    "${PMIX_RANK:-}" \
    "${PMI_RANK:-}" \
    "${MV2_COMM_WORLD_RANK:-}" \
    "${SLURM_PROCID:-}"; do
    if [ -n "${value}" ]; then
      printf '%s\n' "${value}"
      return 0
    fi
  done

  return 1
}

detect_world_size() {
  for value in \
    "${OMPI_COMM_WORLD_SIZE:-}" \
    "${PMIX_SIZE:-}" \
    "${PMI_SIZE:-}" \
    "${MV2_COMM_WORLD_SIZE:-}" \
    "${SLURM_NTASKS:-}"; do
    if [ -n "${value}" ]; then
      printf '%s\n' "${value}"
      return 0
    fi
  done

  return 1
}

under_mpi_launcher() {
  detect_world_rank >/dev/null 2>&1 || detect_world_size >/dev/null 2>&1
}

is_primary_rank() {
  local rank=""
  rank="$(detect_world_rank 2>/dev/null || true)"

  if [ -z "${rank}" ] || [ "${rank}" = "0" ]; then
    return 0
  fi

  return 1
}

log_primary() {
  if is_primary_rank; then
    printf '%s\n' "$*"
  fi
}

source_config_preserving_env() {
  local config_path="$1"
  shift

  if [ ! -f "${config_path}" ]; then
    return 0
  fi

  local -a restore_cmds=()
  local name=""
  local quoted=""
  for name in "$@"; do
    if [ "${!name+x}" = "x" ]; then
      printf -v quoted '%q' "${!name}"
      restore_cmds+=("${name}=${quoted}")
      restore_cmds+=("export ${name}")
    fi
  done

  set -a
  # shellcheck disable=SC1090
  source "${config_path}"
  set +a

  for name in "${restore_cmds[@]}"; do
    eval "${name}"
  done
}

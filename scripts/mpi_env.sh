#!/bin/bash

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
    else
      restore_cmds+=("unset ${name}")
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

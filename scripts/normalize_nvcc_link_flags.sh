#!/bin/bash
set -euo pipefail

out=()

for token in "$@"; do
    case "$token" in
        -Wl,*)
            payload="${token#-Wl,}"
            IFS=',' read -r -a parts <<< "${payload}"
            for part in "${parts[@]}"; do
                if [ -n "${part}" ]; then
                    out+=(-Xlinker "${part}")
                fi
            done
            ;;
        *)
            out+=("${token}")
            ;;
    esac
done

if [ "${#out[@]}" -eq 0 ]; then
    exit 0
fi

printf '%s' "${out[0]}"
for ((i = 1; i < ${#out[@]}; ++i)); do
    printf ' %s' "${out[i]}"
done
printf '\n'

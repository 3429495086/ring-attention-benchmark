#!/bin/bash
# collect_env.sh - Collect system environment information for benchmark reproducibility

set -euo pipefail

echo "========== Ring Attention Benchmark Environment =========="
echo ""

echo "=== System Info ==="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "User: $(whoami)"
echo "Working directory: $(pwd)"
echo ""

echo "=== OS Info ==="
if [ -f /etc/os-release ]; then
    cat /etc/os-release | grep -E "^(NAME|VERSION)=" || true
fi
uname -a
echo ""

echo "=== GPU Info ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version,pci.bus_id --format=csv
    echo ""
    echo "=== GPU Topology ==="
    nvidia-smi topo -m
else
    echo "nvidia-smi not found"
fi
echo ""

echo "=== CUDA Version ==="
if command -v nvcc &> /dev/null; then
    nvcc --version
else
    echo "nvcc not found"
fi
echo ""

echo "=== MPI Version ==="
MPIRUN=""
for candidate in \
    /nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5/bin/mpirun \
    /usr/bin/mpirun \
    mpirun; do
    if command -v "$candidate" &> /dev/null; then
        MPIRUN="$candidate"
        break
    fi
done

if [ -n "$MPIRUN" ]; then
    echo "mpirun path: $MPIRUN"
    "$MPIRUN" --version 2>&1 | head -5
else
    echo "mpirun not found"
fi
echo ""

echo "=== MPI Compiler Wrapper ==="
for candidate in mpicxx mpic++ mpicc; do
    if command -v "$candidate" &> /dev/null; then
        echo "$candidate path: $(command -v "$candidate")"
        "$candidate" --showme:compile 2>/dev/null || "$candidate" -show 2>/dev/null || true
        "$candidate" --showme:link 2>/dev/null || true
        break
    fi
done
echo ""

echo "=== MPI CUDA Support ==="
OMPI_INFO=""
for candidate in \
    /nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5/bin/ompi_info \
    /usr/bin/ompi_info \
    ompi_info; do
    if command -v "$candidate" &> /dev/null; then
        OMPI_INFO="$candidate"
        break
    fi
done

if [ -n "$OMPI_INFO" ]; then
    "$OMPI_INFO" --parsable 2>/dev/null | grep -i cuda || echo "No CUDA support info found"
else
    echo "ompi_info not found"
fi
echo ""

echo "=== NCCL Version ==="
NCCL_HEADER=""
for candidate in \
    /nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/nccl/include/nccl.h \
    /usr/include/nccl.h \
    /usr/local/cuda/include/nccl.h; do
    if [ -f "$candidate" ]; then
        NCCL_HEADER="$candidate"
        break
    fi
done

if [ -n "$NCCL_HEADER" ]; then
    echo "nccl header: $NCCL_HEADER"
    grep -E "NCCL_MAJOR|NCCL_MINOR|NCCL_PATCH" "$NCCL_HEADER" | head -3
else
    echo "nccl.h not found"
fi
echo ""

echo "=== Relevant Environment Variables ==="
env | grep -E '^(CUDA|NCCL|UCX|OMPI|PMI|PMIX|SLURM|MV2|MPI|LD_LIBRARY_PATH)=' | sort || true
echo ""

echo "=== Network Info ==="
if command -v ibstat &> /dev/null; then
    echo "InfiniBand devices:"
    ibstat 2>/dev/null | grep -E "CA|Port|State|Rate" || echo "No IB devices"
else
    echo "ibstat not found (no InfiniBand?)"
fi
if command -v ibv_devinfo &> /dev/null; then
    echo ""
    echo "ibv_devinfo:"
    ibv_devinfo 2>/dev/null | grep -E "hca_id|transport|fw_ver|node_guid|phys_port_cnt|state|active_mtu|active_width|active_speed" || true
fi
if command -v ucx_info &> /dev/null; then
    echo ""
    echo "UCX version:"
    ucx_info -v 2>/dev/null | head -20 || true
fi
echo ""

echo "=== CPU Info ==="
if [ -f /proc/cpuinfo ]; then
    grep -m1 "model name" /proc/cpuinfo || true
    echo "CPU cores: $(grep -c processor /proc/cpuinfo)"
fi
echo ""

echo "=== Memory Info ==="
if command -v free &> /dev/null; then
    free -h | head -2
fi
echo ""

echo "========== End Environment Info =========="

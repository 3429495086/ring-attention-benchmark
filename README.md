# Ring Attention Multi-GPU Benchmark

Ring Attention implementation with multiple communication backends for benchmarking multi-GPU performance.

## Benchmarks

| Type | Description |
|------|-------------|
| **comm** (loop) | Communication-only benchmark, no attention computation |
| **attention** | Full Ring Attention with online softmax merge |

## Communication Backends

| Backend | Description |
|---------|-------------|
| `staged` | Host-staged MPI (GPU→CPU→MPI→CPU→GPU) with `MPI_Sendrecv` |
| `staged_Isendrecv` | Host-staged MPI with `MPI_Isend/Irecv` |
| `cuda_aware` | CUDA-aware MPI (direct GPU pointers) with `MPI_Sendrecv` |
| `cuda_aware_Isendrecv` | CUDA-aware MPI with `MPI_Isend/Irecv` |
| `nccl` | NVIDIA NCCL with `ncclSend/ncclRecv` |

## Dependencies

- CUDA Toolkit (tested with 12.x)
- MPI implementation with CUDA-aware support (e.g., OpenMPI 4.x with UCX)
- NCCL (for NCCL backend)

## Quick Start

```bash
# 1. Optional: create a machine-specific config
cp config.example.env benchmark.env
# edit benchmark.env if CUDA/MPI/NCCL are not already in PATH/LD_LIBRARY_PATH

# 2. Build all benchmarks
make all

# 3. Run the full reproducible suite
./run_suite.sh
```

Results are written to `results/<hostname>_<timestamp>/` with:

- `manifest.txt`: run configuration
- `build.log`: build output
- `env_local.txt`: hardware/software environment
- `comm_np<N>.log`: communication-only results
- `attention_np<N>.log`: full Ring Attention results

Manual run:

```bash
./collect_env.sh > env_info.txt
./benchmark_comm.sh > results_comm.txt 2>&1
./benchmark_attention.sh > results_attention.txt 2>&1
```

## Recommended Test Matrix

The default `config.example.env` is the recommended starting point for comparing machines:

```bash
NP_LIST="2 4 8"
COMM_SIZES="262144 524288 1048576 4194304 16777216"
ATTENTION_SIZES="262144 524288 1048576 4194304"
BACKENDS="staged staged_Isendrecv cuda_aware cuda_aware_Isendrecv nccl"
WARMUP=10
ITERS=100
RUN_TYPES="comm attention"
```

This matrix compares:

| Comparison | Purpose |
|------------|---------|
| Staged MPI vs CUDA-aware MPI vs NCCL | Measure whether GPU-direct paths, NVLink, or InfiniBand help on a given machine |
| Different `NP_LIST` values | Show how communication cost changes as GPU/rank count grows |
| Different message sizes | Separate latency-dominated small transfers from bandwidth-dominated large transfers |
| `comm` vs `attention` | Estimate how much of the full Ring Attention time is communication |

Adjust `NP_LIST` to match the available GPUs. For example, use `NP_LIST="2 4"` on a 4-GPU node, or `NP_LIST="4 8"` on two 4-GPU nodes.

## Slurm Clusters

Inside a Slurm allocation, `run_suite.sh` now defaults to `srun` for benchmark launches and avoids the nested `srun -> mpirun` problem. Submit a batch script, then call `./run_suite.sh` directly from inside the batch job.

Use `batch_slurm.example.sh` as a starting point:

```bash
sbatch batch_slurm.example.sh
```

Minimal pattern:

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1

module load nvhpc/24.5
module load cuda/12.4
module load openmpi/4.1.7-nvhpc24.5
module load slurm/17.11.5

export LAUNCHER=srun
export SRUN_MPI_TYPE=openmpi
export NP_LIST="2"

./run_suite.sh
```

Important:

- Start `run_suite.sh` directly inside `sbatch`.
- Do not use `srun ./run_suite.sh`.
- Let `run_suite.sh` create the internal `srun -n <NP>` steps for each benchmark run.

## Running On Another Machine

Send the repository link and ask the operator to run:

```bash
git clone https://github.com/YOUR_USERNAME/ring-attention-benchmark.git
cd ring-attention-benchmark

cp config.example.env benchmark.env
# Edit benchmark.env if CUDA, MPI, NCCL, Slurm, hostfile, or launcher flags are machine-specific.

make print-config
make all
./run_suite.sh
```

Ask them to send back the generated `results/<hostname>_<timestamp>/` directory. If the machine has NVLink or InfiniBand, the most important logs are usually `comm_np<N>.log` for `cuda_aware`, `cuda_aware_Isendrecv`, and `nccl`.

Example Russian note:

```text
Здравствуйте,

Это бенчмарк Ring Attention для multi-GPU. Он сравнивает staged MPI,
CUDA-aware MPI и NCCL для разных размеров данных и числа GPU.

Пожалуйста, запустите:

git clone https://github.com/YOUR_USERNAME/ring-attention-benchmark.git
cd ring-attention-benchmark
make print-config
make all
./run_suite.sh

Результаты будут сохранены в results/<hostname>_<timestamp>/.
Пожалуйста, пришлите мне эту папку.
```

## Build

```bash
make all        # Build both comm and attention
make comm       # Build communication-only benchmarks
make attention  # Build attention benchmarks
make print-config # Show detected compiler/library flags
make clean      # Remove built files
make help       # Show help
```

The Makefile uses `nvcc`, `mpicxx`, and `-lnccl` from the current environment by default.
Override paths when a cluster keeps CUDA/MPI/NCCL in non-standard locations:

```bash
CUDA_HOME=/usr/local/cuda \
MPI_HOME=/opt/openmpi \
NCCL_HOME=/opt/nccl \
make all
```

## Run

### Configuration

Environment variables for `run_suite.sh`:

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG` | `benchmark.env` | Optional config file to source |
| `RUN_LABEL` | `<hostname>_<timestamp>` | Result directory name |
| `LAUNCHER` | `auto` | `auto`, `mpirun`, or `srun` |
| `NP_LIST` | `2` | MPI process counts to sweep |
| `RUN_TYPES` | `comm attention` | Which benchmark families to run |
| `BUILD` | `1` | Run `make all` before benchmarking |
| `COLLECT_ENV` | `1` | Save `collect_env.sh` output |
| `COMM_SIZES` | see below | KV sizes for communication benchmarks |
| `ATTENTION_SIZES` | see below | KV sizes for attention benchmarks |

Environment variables for `benchmark_comm.sh` and `benchmark_attention.sh`:

| Variable | Default | Description |
|----------|---------|-------------|
| `NP` | 2 | Number of MPI processes (= number of GPUs) |
| `WARMUP` | 10 | Warmup iterations (not timed) |
| `ITERS` | 100 | Timed iterations |
| `SIZES` | see below | KV shard sizes in bytes |
| `BACKENDS` | all backends | Space-separated backend list |
| `MPIRUN` | auto-detected | Path to mpirun |
| `HOSTFILE` | unset | OpenMPI-style hostfile path |
| `MPIRUN_EXTRA_ARGS` | unset | Extra mpirun flags, e.g. mapping/binding |
| `SRUN` | auto-detected | Path to srun |
| `SRUN_MPI_TYPE` | auto-detected | Slurm MPI plugin, e.g. `openmpi` |
| `SRUN_EXTRA_ARGS` | unset | Extra srun flags, e.g. `--ntasks-per-node=1` |
| `ALLOW_GPU_OVERSUBSCRIBE` | unset | Set to `1` only when multiple ranks may share one GPU |

Default sizes:
- **comm**: `"262144 524288 1048576 4194304 16777216"` (256KB to 16MB)
- **attention**: `"262144 524288 1048576 4194304"` (256KB to 4MB)

### Examples

```bash
# Run communication benchmark with 4 GPUs
NP=4 ./benchmark_comm.sh

# Run only NCCL and CUDA-aware MPI
BACKENDS="cuda_aware nccl" NP=4 ./benchmark_comm.sh

# Run attention benchmark with larger sizes
NP=2 SIZES="1048576 4194304 16777216" ./benchmark_attention.sh

# Multi-node OpenMPI example: 4 GPUs per node, 2 nodes, 8 ranks total
NP_LIST="4 8" \
HOSTFILE=hosts.txt \
MPIRUN_EXTRA_ARGS="--map-by ppr:4:node --bind-to none" \
./run_suite.sh

# Slurm example: two nodes, one GPU task on each node
LAUNCHER=srun \
SRUN_MPI_TYPE=openmpi \
SRUN_EXTRA_ARGS="--ntasks-per-node=1" \
NP_LIST="2" \
./run_suite.sh

# Run individual executable
mpirun -np 2 ./bin/ring_loop_nccl 1048576 10 100
mpirun -np 2 ./bin/ring_attention_nccl_gpu_bench 1048576 10 100
```

### Parameters

```
./bin/ring_loop_* <kv_size_bytes> [warmup] [iters]
./bin/ring_attention_* <kv_size_bytes> [warmup] [iters]
```

- `kv_size_bytes`: Size of KV shard per GPU in bytes
  - For attention: must be multiple of `head_dim * sizeof(float)` = 256 bytes
- `warmup`: Warmup iterations (default: 10)
- `iters`: Timed iterations (default: 100)

| kv_size_bytes | seq_q = seq_k (attention) |
|---------------|---------------------------|
| 262144 (256 KB) | 1024 |
| 524288 (512 KB) | 2048 |
| 1048576 (1 MB) | 4096 |
| 4194304 (4 MB) | 16384 |
| 16777216 (16 MB) | 65536 |

## Output Format

### Communication (loop) benchmark

```
RESULT backend=staged bytes=262144 ranks=2 warmup=10 iters=100 avg_d2h_ms=0.027 avg_mpi_ms=0.080 avg_h2d_ms=0.028 avg_total_ms=0.135 total_GBps=1.94 mpi_GBps=3.28
```

| Field | Description |
|-------|-------------|
| `avg_d2h_ms` | Device-to-Host copy time (staged only) |
| `avg_mpi_ms` / `avg_nccl_ms` | Communication time |
| `avg_h2d_ms` | Host-to-Device copy time (staged only) |
| `avg_total_ms` | Total loop time |
| `total_GBps` | End-to-end bandwidth |
| `mpi_GBps` | MPI/NCCL-only bandwidth |

### Attention benchmark

```
staged bench: warmup=10 iters=100
avg_total=24.677 ms avg_D2H=0.027 avg_MPI=0.080 avg_H2D=0.028 avg_attn=24.218 avg_merge=0.079 avg_final=0.269
```

| Field | Description |
|-------|-------------|
| `avg_total` | Total time per iteration (ms) |
| `avg_D2H` | Device-to-Host copy time (staged only) |
| `avg_MPI` / `avg_NCCL` | Communication time |
| `avg_H2D` | Host-to-Device copy time (staged only) |
| `avg_attn` | GPU attention kernel time |
| `avg_merge` | Online softmax merge kernel time |
| `avg_final` | Final normalization + D2H copy time |

## File Structure

```
ring_attention_benchmark/
├── Makefile
├── README.md
├── config.example.env
├── batch_slurm.example.sh
├── run_suite.sh
├── benchmark_comm.sh
├── benchmark_attention.sh
├── collect_env.sh
├── src/
│   ├── common/
│   │   └── device_utils.cuh
│   ├── loop/
│   │   ├── ring_loop_staged.cu
│   │   ├── ring_loop_staged_Isendrecv.cu
│   │   ├── ring_loop_cuda_aware.cu
│   │   ├── ring_loop_cuda_aware_Isendrecv.cu
│   │   └── ring_loop_nccl.cu
│   └── attention/
│       ├── ring_attention_staged_gpu_bench.cu
│       ├── ring_attention_staged_Isendrecv_gpu_bench.cu
│       ├── ring_attention_cuda_aware_gpu_bench.cu
│       ├── ring_attention_cuda_aware_Isendrecv_gpu_bench.cu
│       └── ring_attention_nccl_gpu_bench.cu
├── bin/                  # Built executables
└── results/              # Output logs
```

## Notes

- Each MPI rank binds to GPU by local rank on its node. The code checks common MPI launcher variables first (`OMPI_COMM_WORLD_LOCAL_RANK`, `SLURM_LOCALID`, `MV2_COMM_WORLD_LOCAL_RANK`, etc.) and falls back to `MPI_Comm_split_type`.
- By default the benchmark aborts if ranks per node exceed visible GPUs. Set `ALLOW_GPU_OVERSUBSCRIBE=1` only for deliberate oversubscription tests.
- On Slurm clusters, launch the suite from `sbatch` with `./run_suite.sh`. The internal benchmark steps can then use `srun` safely.
- Ensure no other jobs are running on the GPUs during benchmarks
- For CUDA-aware MPI: requires MPI built with CUDA support
- For best NCCL performance: use NVLink or InfiniBand interconnect
- For clean comparison, run the same `config.example.env` settings on each machine and keep the generated result directory.

## Author

NNiang - Ring Attention 多GPU并行研究

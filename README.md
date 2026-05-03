# Ring Attention Multi-GPU Benchmark

Ring Attention implementation with five communication backends and two overlap-enabled attention variants:

- `staged`
- `staged_Isendrecv`
- `staged_Isendrecv_overlap` (attention benchmark only)
- `cuda_aware`
- `cuda_aware_Isendrecv`
- `cuda_aware_Isendrecv_overlap` (attention benchmark only)
- `nccl`

The package now follows an external-launch model:

- `make` reads `benchmark.env` directly
- `run_suite.sh` only prepares the run directory and build artifacts
- `benchmark_comm.sh` and `benchmark_attention.sh` launch the benchmark binaries directly with `mpirun` or `srun`
- `launch_mpirun.example.sh` and `batch_slurm.example.sh` are ready-made outer launchers

## Benchmarks

| Type | Description |
|------|-------------|
| `comm` | Communication-only benchmark |
| `attention` | Full Ring Attention benchmark |

## Dependencies

- CUDA Toolkit
- MPI implementation with GPU support
- NCCL for the NCCL backend

## Quick Start

```bash
cp config.example.env benchmark.env
make print-config
./run_suite.sh
```

`./run_suite.sh` in this mode does not launch MPI jobs. It:

1. creates `results/<hostname>_<timestamp>/`
2. builds the binaries
3. collects local environment info
4. writes `launch_examples.txt` with the exact `mpirun` and `srun` commands

Then launch the actual MPI jobs externally.

## Build Configuration

`make` automatically loads `benchmark.env` when it exists, so this works:

```bash
cp config.example.env benchmark.env
# edit CUDA_HOME / MPI_HOME / NCCL_HOME if needed
make print-config
make all
```

If `MPI_HOME` or `NCCL_HOME` is set, the Makefile now prefers those explicit paths over incomplete wrapper autodetection. This is useful on clusters where plain `mpicxx --showme:*` does not return enough information.

Example:

```bash
MPI_HOME=/opt/openmpi \
NCCL_HOME=/opt/nccl \
make print-config
```

Useful targets:

```bash
make all
make comm
make attention
make print-config
make clean
```

## Recommended Test Matrix

The default `config.example.env` is a good starting point:

```bash
NP_LIST="2 4 8"
COMM_SIZES="262144 524288 1048576 4194304 16777216"
ATTENTION_SIZES="262144 524288 1048576"
BACKENDS="staged staged_Isendrecv cuda_aware cuda_aware_Isendrecv nccl"
ATTENTION_BACKENDS="staged staged_Isendrecv staged_Isendrecv_overlap cuda_aware cuda_aware_Isendrecv cuda_aware_Isendrecv_overlap nccl"
WARMUP=10
ITERS=100
RUN_TYPES="comm attention"
```

This covers:

- different rank/GPU counts
- small and large message sizes
- staged MPI vs CUDA-aware MPI vs NCCL
- pure communication vs full attention
- overlap-enabled attention variants without changing the comm-only suite

## External Launch Model

### Preparation Step

Run once from the repository root:

```bash
./run_suite.sh
```

This writes:

- `manifest.txt`
- `build.log`
- `env_local.txt`
- `run_context.env`
- `launch_examples.txt`

### Launch Step

`run_suite.sh` is preparation-only. The actual launch step should call
`benchmark_comm.sh` or `benchmark_attention.sh` from a normal shell. Those
wrappers call `mpirun` or `srun` directly on the benchmark binaries.

## Ready-Made Launcher Examples

### OpenMPI

Use the external helper:

```bash
bash launch_mpirun.example.sh
```

It will:

1. call `./run_suite.sh` once for preparation
2. loop over `NP_LIST`
3. call `benchmark_comm.sh` / `benchmark_attention.sh`
4. let those wrappers directly launch the benchmark binaries with `mpirun`
5. write logs into the prepared results directory

### Slurm

Use:

```bash
sbatch batch_slurm.example.sh
```

The batch script now does the launching itself:

1. prepare with `./run_suite.sh`
2. loop over `NP_LIST`
3. call `benchmark_comm.sh` / `benchmark_attention.sh`
4. let those wrappers directly launch the benchmark binaries with `srun`

This matches the cluster requirement that the launcher is outside the benchmark scripts.

## Manual Examples

### OpenMPI

```bash
./run_suite.sh
RUN_LABEL="$(hostname)_20260421_test"
RESULTS_ROOT="$PWD/results"

env CONFIG=benchmark.env NP=2 RUN_LABEL="$RUN_LABEL" RESULTS_ROOT="$RESULTS_ROOT" \
  LAUNCHER=mpirun ./benchmark_comm.sh > "$RESULTS_ROOT/$RUN_LABEL/comm_np2.log" 2>&1

env CONFIG=benchmark.env NP=2 RUN_LABEL="$RUN_LABEL" RESULTS_ROOT="$RESULTS_ROOT" \
  LAUNCHER=mpirun ./benchmark_attention.sh > "$RESULTS_ROOT/$RUN_LABEL/attention_np2.log" 2>&1
```

### Slurm

```bash
./run_suite.sh
env CONFIG=benchmark.env NP=2 RUN_LABEL="$RUN_LABEL" RESULTS_ROOT="$RESULTS_ROOT" \
  LAUNCHER=srun SRUN_MPI_TYPE=openmpi SRUN_EXTRA_ARGS="--ntasks-per-node=1" \
  ./benchmark_comm.sh > "$RESULTS_ROOT/$RUN_LABEL/comm_np2.log" 2>&1
```

### Direct wrapper launch

You can also launch the benchmark family wrappers directly:

```bash
env NP=2 LAUNCHER=mpirun ./benchmark_comm.sh
env NP=2 LAUNCHER=mpirun ./benchmark_attention.sh
```

## Attention Correctness Check

Use this before collecting performance results on a new machine:

```bash
env NP=2 SIZE=262144 WARMUP=1 ITERS=1 LAUNCHER=mpirun \
  ./check_attention_correctness.sh
```

The script compares:

- `staged_Isendrecv` vs `staged_Isendrecv_overlap`
- `cuda_aware_Isendrecv` vs `cuda_aware_Isendrecv_overlap`

It temporarily enables `ATTENTION_DUMP_OUTPUT=1`, runs the baseline and overlap
versions, saves `ring_output_rank*.bin` under `results/<run_label>/`, and prints
`PASS` or `FAIL` with `max_abs_err`, `mean_abs_err`, and `max_rel_err`.

You can also run it through Make:

```bash
make check-attention-correctness
```

## Main Variables

### Preparation

| Variable | Default | Meaning |
|----------|---------|---------|
| `CONFIG` | `benchmark.env` | Config file to source |
| `RUN_LABEL` | `<hostname>_<timestamp>` | Result directory name |
| `RESULTS_ROOT` | `results/` | Root directory for outputs |
| `BUILD` | `1` | Build binaries during preparation |
| `COLLECT_ENV` | `1` | Save `collect_env.sh` output |

### Benchmark Matrix

| Variable | Default | Meaning |
|----------|---------|---------|
| `NP_LIST` | `2` | Rank counts to sweep from the outer launcher |
| `RUN_TYPES` | `comm attention` | Benchmark families |
| `BACKENDS` | five comm backends | Communication backends for `benchmark_comm.sh` |
| `ATTENTION_BACKENDS` | seven attention backends | Attention backends for `benchmark_attention.sh` |
| `COMM_SIZES` | see config | Sizes for communication benchmark |
| `ATTENTION_SIZES` | see config | Sizes for attention benchmark |
| `WARMUP` | `10` | Warmup iterations |
| `ITERS` | `100` | Timed iterations |

### External Launcher Hints

These variables are used by the wrapper scripts that launch the binaries directly.

| Variable | Meaning |
|----------|---------|
| `MPIRUN` | Path to `mpirun` |
| `HOSTFILE` | OpenMPI hostfile |
| `MPIRUN_EXTRA_ARGS` | Extra `mpirun` flags |
| `SRUN` | Path to `srun` |
| `SRUN_MPI_TYPE` | Slurm MPI plugin, for example `openmpi` |
| `SRUN_EXTRA_ARGS` | Extra `srun` flags |

## Output

The main result directory contains:

- `manifest.txt`
- `build.log`
- `env_local.txt`
- `run_context.env`
- `launch_examples.txt`
- `comm_np<N>.log`
- `attention_np<N>.log`

Example communication output:

```text
RESULT backend=staged bytes=262144 ranks=2 warmup=10 iters=100 avg_d2h_ms=0.027 avg_mpi_ms=0.080 avg_h2d_ms=0.028 avg_total_ms=0.135 total_GBps=1.94 mpi_GBps=3.28
```

Example attention output:

```text
staged bench: warmup=10 iters=100
avg_total=24.677 ms avg_D2H=0.027 avg_MPI=0.080 avg_H2D=0.028 avg_attn=24.218 avg_merge=0.079 avg_final=0.269
```

## Notes

- GPU selection inside the CUDA code is based on local rank, not global rank.
- `benchmark_comm.sh` and `benchmark_attention.sh` print headers only from rank 0 to keep logs readable.
- If `NP` differs from the detected MPI world size, the detected world size wins.

Yu Gang - Ring Attention Multi-GPU Parallelization Research

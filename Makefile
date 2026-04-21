# Ring Attention Benchmark - Makefile
# Supports: staged MPI, staged MPI Isend/Irecv, CUDA-aware MPI, CUDA-aware MPI Isend/Irecv, NCCL

# ============ Configuration (override for your system if needed) ============
CUDA_HOME    ?=
MPI_HOME     ?=
NCCL_HOME    ?=

# ============ Compilers and flags ============
NVCC         ?= $(if $(CUDA_HOME),$(CUDA_HOME)/bin/nvcc,nvcc)
MPICXX       ?= mpicxx
NVCCFLAGS    ?= -O3
NORMALIZE_NVCC_LINK_FLAGS := ./scripts/normalize_nvcc_link_flags.sh

RAW_MPI_INC  := $(strip $(shell $(MPICXX) --showme:compile 2>/dev/null))
RAW_MPI_LIB  := $(strip $(shell $(MPICXX) --showme:link 2>/dev/null || echo -lmpi))

DEFAULT_MPI_INC := $(if $(RAW_MPI_INC),$(RAW_MPI_INC),$(if $(MPI_HOME),-I$(MPI_HOME)/include,))
DEFAULT_MPI_LIB := $(if $(RAW_MPI_LIB),$(shell $(NORMALIZE_NVCC_LINK_FLAGS) $(RAW_MPI_LIB)),$(if $(MPI_HOME),-L$(MPI_HOME)/lib -lmpi,-lmpi))

MPI_INC      ?= $(DEFAULT_MPI_INC)
MPI_LIB      ?= $(DEFAULT_MPI_LIB)

NCCL_INC     ?= $(if $(NCCL_HOME),-I$(NCCL_HOME)/include,)
NCCL_LIB     ?= $(if $(NCCL_HOME),-L$(NCCL_HOME)/lib -lnccl,-lnccl)

# ============ Directories ============
SRC_ATTN     := src/attention
SRC_LOOP     := src/loop
COMMON_HEADERS := src/common/device_utils.cuh
BIN_DIR      := bin

# ============ Targets ============
COMM_EXES := \
	$(BIN_DIR)/ring_loop_staged \
	$(BIN_DIR)/ring_loop_staged_Isendrecv \
	$(BIN_DIR)/ring_loop_cuda_aware \
	$(BIN_DIR)/ring_loop_cuda_aware_Isendrecv \
	$(BIN_DIR)/ring_loop_nccl

ATTENTION_EXES := \
	$(BIN_DIR)/ring_attention_staged_gpu_bench \
	$(BIN_DIR)/ring_attention_staged_Isendrecv_gpu_bench \
	$(BIN_DIR)/ring_attention_cuda_aware_gpu_bench \
	$(BIN_DIR)/ring_attention_cuda_aware_Isendrecv_gpu_bench \
	$(BIN_DIR)/ring_attention_nccl_gpu_bench

.PHONY: all comm attention clean help print-config

all: comm attention
	@echo "=== All benchmarks built ==="

comm: $(BIN_DIR) $(COMM_EXES)
	@echo "=== Communication benchmarks built ==="

attention: $(BIN_DIR) $(ATTENTION_EXES)
	@echo "=== Attention benchmarks built ==="

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

print-config:
	@echo "NVCC=$(NVCC)"
	@echo "MPICXX=$(MPICXX)"
	@echo "NVCCFLAGS=$(NVCCFLAGS)"
	@echo "MPI_INC=$(MPI_INC)"
	@echo "MPI_LIB=$(MPI_LIB)"
	@echo "NCCL_INC=$(NCCL_INC)"
	@echo "NCCL_LIB=$(NCCL_LIB)"

# ============ Communication (loop) benchmarks ============
$(BIN_DIR)/ring_loop_staged: $(SRC_LOOP)/ring_loop_staged.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(MPI_INC) $(MPI_LIB)

$(BIN_DIR)/ring_loop_staged_Isendrecv: $(SRC_LOOP)/ring_loop_staged_Isendrecv.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(MPI_INC) $(MPI_LIB)

$(BIN_DIR)/ring_loop_cuda_aware: $(SRC_LOOP)/ring_loop_cuda_aware.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(MPI_INC) $(MPI_LIB)

$(BIN_DIR)/ring_loop_cuda_aware_Isendrecv: $(SRC_LOOP)/ring_loop_cuda_aware_Isendrecv.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(MPI_INC) $(MPI_LIB)

$(BIN_DIR)/ring_loop_nccl: $(SRC_LOOP)/ring_loop_nccl.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(MPI_INC) $(MPI_LIB) $(NCCL_INC) $(NCCL_LIB)

# ============ Attention benchmarks ============
$(BIN_DIR)/ring_attention_staged_gpu_bench: $(SRC_ATTN)/ring_attention_staged_gpu_bench.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(MPI_INC) $(MPI_LIB)

$(BIN_DIR)/ring_attention_staged_Isendrecv_gpu_bench: $(SRC_ATTN)/ring_attention_staged_Isendrecv_gpu_bench.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(MPI_INC) $(MPI_LIB)

$(BIN_DIR)/ring_attention_cuda_aware_gpu_bench: $(SRC_ATTN)/ring_attention_cuda_aware_gpu_bench.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(MPI_INC) $(MPI_LIB)

$(BIN_DIR)/ring_attention_cuda_aware_Isendrecv_gpu_bench: $(SRC_ATTN)/ring_attention_cuda_aware_Isendrecv_gpu_bench.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(MPI_INC) $(MPI_LIB)

$(BIN_DIR)/ring_attention_nccl_gpu_bench: $(SRC_ATTN)/ring_attention_nccl_gpu_bench.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(MPI_INC) $(MPI_LIB) $(NCCL_INC) $(NCCL_LIB)

clean:
	rm -rf $(BIN_DIR)
	rm -f ring_output_rank*.bin

help:
	@echo "Ring Attention Benchmark Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make all        - Build all benchmarks (default)"
	@echo "  make comm       - Build communication-only benchmarks (loop)"
	@echo "  make attention  - Build full attention benchmarks"
	@echo "  make print-config - Show resolved compiler and library flags"
	@echo "  make clean      - Remove all built files"
	@echo "  make help       - Show this message"
	@echo ""
	@echo "Configuration (override with environment or command line):"
	@echo "  CUDA_HOME=$(CUDA_HOME)"
	@echo "  MPI_HOME=$(MPI_HOME)"
	@echo "  NCCL_HOME=$(NCCL_HOME)"
	@echo "  NVCC=$(NVCC)"
	@echo "  MPICXX=$(MPICXX)"
	@echo ""
	@echo "Example:"
	@echo "  CUDA_HOME=/usr/local/cuda MPI_HOME=/opt/openmpi NCCL_HOME=/opt/nccl make all"

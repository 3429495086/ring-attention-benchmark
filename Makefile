# Ring Attention Benchmark - Makefile
# Supports: 5 communication backends plus overlap attention benchmarks

CONFIG ?= benchmark.env
-include $(CONFIG)

# ============ Configuration (override for your system if needed) ============
CUDA_HOME    ?=
MPI_HOME     ?=
NCCL_HOME    ?=

# ============ Compilers and flags ============
NVCC         ?= $(if $(CUDA_HOME),$(CUDA_HOME)/bin/nvcc,nvcc)
MPICXX       ?= mpicxx
NVCCFLAGS    ?= -O3
NORMALIZE_NVCC_LINK_FLAGS := /bin/sh ./scripts/normalize_nvcc_link_flags.sh

strip-trailing-slash = $(patsubst %/,%,$(1))

MPI_HOME_CLEAN  := $(call strip-trailing-slash,$(MPI_HOME))
NCCL_HOME_CLEAN := $(call strip-trailing-slash,$(NCCL_HOME))

MPI_INC_DIR := $(if $(MPI_HOME_CLEAN),$(if $(wildcard $(MPI_HOME_CLEAN)/include),$(MPI_HOME_CLEAN)/include,$(MPI_HOME_CLEAN)/include),)
MPI_LIB_DIR := $(if $(MPI_HOME_CLEAN),$(if $(wildcard $(MPI_HOME_CLEAN)/lib),$(MPI_HOME_CLEAN)/lib,$(if $(wildcard $(MPI_HOME_CLEAN)/lib64),$(MPI_HOME_CLEAN)/lib64,$(MPI_HOME_CLEAN)/lib)),)
MPI_CXX_LIB_PRESENT := $(if $(MPI_LIB_DIR),$(strip $(wildcard $(MPI_LIB_DIR)/libmpi_cxx.*)),)

NCCL_INC_DIR := $(if $(NCCL_HOME_CLEAN),$(if $(wildcard $(NCCL_HOME_CLEAN)/include),$(NCCL_HOME_CLEAN)/include,$(NCCL_HOME_CLEAN)/include),)
NCCL_LIB_DIR := $(if $(NCCL_HOME_CLEAN),$(if $(wildcard $(NCCL_HOME_CLEAN)/lib),$(NCCL_HOME_CLEAN)/lib,$(if $(wildcard $(NCCL_HOME_CLEAN)/lib64),$(NCCL_HOME_CLEAN)/lib64,$(NCCL_HOME_CLEAN)/lib)),)

RAW_MPI_INC  := $(strip $(shell $(MPICXX) --showme:compile 2>/dev/null))
RAW_MPI_LIB  := $(strip $(shell $(MPICXX) --showme:link 2>/dev/null))
NORMALIZED_RAW_MPI_LIB := $(if $(RAW_MPI_LIB),$(shell $(NORMALIZE_NVCC_LINK_FLAGS) $(RAW_MPI_LIB)),)

DEFAULT_MPI_INC := $(if $(MPI_HOME_CLEAN),-I$(MPI_INC_DIR),$(RAW_MPI_INC))
DEFAULT_MPI_LIB := $(if $(MPI_HOME_CLEAN),-L$(MPI_LIB_DIR) $(if $(MPI_CXX_LIB_PRESENT),-lmpi_cxx,) -lmpi,$(if $(NORMALIZED_RAW_MPI_LIB),$(NORMALIZED_RAW_MPI_LIB),-lmpi))

MPI_INC      ?= $(DEFAULT_MPI_INC)
MPI_LIB      ?= $(DEFAULT_MPI_LIB)

NCCL_INC     ?= $(if $(NCCL_HOME_CLEAN),-I$(NCCL_INC_DIR),)
NCCL_LIB     ?= $(if $(NCCL_HOME_CLEAN),-L$(NCCL_LIB_DIR) -lnccl,-lnccl)

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
	$(BIN_DIR)/ring_attention_staged_Isendrecv_overlap_gpu_bench \
	$(BIN_DIR)/ring_attention_cuda_aware_gpu_bench \
	$(BIN_DIR)/ring_attention_cuda_aware_Isendrecv_gpu_bench \
	$(BIN_DIR)/ring_attention_cuda_aware_Isendrecv_overlap_gpu_bench \
	$(BIN_DIR)/ring_attention_nccl_gpu_bench

.PHONY: all comm attention check-attention-correctness clean help print-config

all: comm attention
	@echo "=== All benchmarks built ==="

comm: $(BIN_DIR) $(COMM_EXES)
	@echo "=== Communication benchmarks built ==="

attention: $(BIN_DIR) $(ATTENTION_EXES)
	@echo "=== Attention benchmarks built ==="

check-attention-correctness:
	./check_attention_correctness.sh

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(COMM_EXES) $(ATTENTION_EXES): | $(BIN_DIR)

print-config:
	@echo "CONFIG=$(CONFIG)"
	@echo "NVCC=$(NVCC)"
	@echo "MPICXX=$(MPICXX)"
	@echo "NVCCFLAGS=$(NVCCFLAGS)"
	@echo "MPI_HOME=$(MPI_HOME)"
	@echo "MPI_INC=$(MPI_INC)"
	@echo "MPI_LIB=$(MPI_LIB)"
	@echo "NCCL_HOME=$(NCCL_HOME)"
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

$(BIN_DIR)/ring_attention_staged_Isendrecv_overlap_gpu_bench: $(SRC_ATTN)/ring_attention_staged_Isendrecv_overlap_gpu_bench.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(MPI_INC) $(MPI_LIB)

$(BIN_DIR)/ring_attention_cuda_aware_gpu_bench: $(SRC_ATTN)/ring_attention_cuda_aware_gpu_bench.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(MPI_INC) $(MPI_LIB)

$(BIN_DIR)/ring_attention_cuda_aware_Isendrecv_gpu_bench: $(SRC_ATTN)/ring_attention_cuda_aware_Isendrecv_gpu_bench.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(MPI_INC) $(MPI_LIB)

$(BIN_DIR)/ring_attention_cuda_aware_Isendrecv_overlap_gpu_bench: $(SRC_ATTN)/ring_attention_cuda_aware_Isendrecv_overlap_gpu_bench.cu $(COMMON_HEADERS)
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
	@echo "  make check-attention-correctness - Compare baseline and overlap attention outputs"
	@echo "  make print-config - Show resolved compiler and library flags"
	@echo "  make clean      - Remove all built files"
	@echo "  make help       - Show this message"
	@echo ""
	@echo "Configuration (auto-loads benchmark.env when present):"
	@echo "  CONFIG=$(CONFIG)"
	@echo "  CUDA_HOME=$(CUDA_HOME)"
	@echo "  MPI_HOME=$(MPI_HOME)"
	@echo "  NCCL_HOME=$(NCCL_HOME)"
	@echo "  NVCC=$(NVCC)"
	@echo "  MPICXX=$(MPICXX)"
	@echo ""
	@echo "Example:"
	@echo "  cp config.example.env benchmark.env"
	@echo "  make print-config"
	@echo "  make all"

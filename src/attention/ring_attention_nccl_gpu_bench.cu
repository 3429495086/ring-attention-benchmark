/*
 * ring_attention_nccl_gpu_bench.cu
 * Ring Attention benchmark — NCCL + GPU attention
 *
 * Compile:
 *   nvcc -o ring_attention_nccl_gpu_bench ring_attention_nccl_gpu_bench.cu \
 *   -I/nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5/include \
 *   -L/nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5/lib \
 *   -lmpi \
 *   -I/nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/nccl/include \
 *   -L/nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/nccl/lib \
 *   -lnccl
 *
 * Run:
 *   /nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5/bin/mpirun \
 *   -np 2 ./ring_attention_nccl_gpu_bench 262144 10 50
 */
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<float.h>
#include<mpi.h>
#include<cuda_runtime.h>
#include<nccl.h>
#include "../common/device_utils.cuh"

// CHECK_CUDA: if any CUDA call fails, stop all processes 
#define CHECK_CUDA(call) do{ \
    cudaError_t err = (call); \
    if(err != cudaSuccess) { \
        fprintf(stderr, "[Rank %d] CUDA error at %s:%d: %s\n", \
                rank, __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

#define CHECK_NCCL(call) do{ \
    ncclResult_t err = (call); \
    if(err != ncclSuccess){ \
        fprintf(stderr, "[Rank %d] NCCL error at %s:%d: %s\n", \
                rank, __FILE__, __LINE__, ncclGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

__global__
void gpu_attention_partial_kernel(
    const float *Q, const float *K, const float *V,
    int seq_q, int seq_k, int dim,
    float *Out, float *m_out, float *lout
){
    int i = blockIdx.x;             // which row query
    int tid = threadIdx.x;          // block thread id
    int blockSize = blockDim.x;     // 256
    if(i >= seq_q)
        return;

    // calculate score, find max
    float local_max = -FLT_MAX;
    for(int j = tid; j < seq_k; j += blockSize){
        float score = 0.0f;
        for(int d = 0; d < dim; d++)
            score += Q[i * dim + d] * K[j * dim + d];
        if(local_max < score)
            local_max = score;
    }

    __shared__ float smax[256];
    smax[tid] = local_max;
    __syncthreads();            // wait all threads
    // reduction: find global max
    for(int stride = blockSize / 2; stride > 0; stride /= 2){
        if(tid < stride && smax[tid] < smax[tid + stride]) 
            smax[tid] = smax[tid + stride];
        __syncthreads();
    }
    float row_max = smax[0];

    // clear Out
    if(tid == 0){
        for(int d = 0; d < dim; d++)
            Out[i * dim + d] = 0.0f;
    }
    __syncthreads();

    // calculate both exp sum and weighted V
    float local_sum = 0.0f;
    for(int j = tid; j < seq_k; j += blockSize){
        float score = 0.0f;
        for(int d = 0; d < dim; d++)
            score += Q[i * dim + d] * K[j * dim + d];
        float w = expf(score - row_max);
        local_sum += w;
        for(int d = 0; d < dim; d++)
            atomicAdd(&Out[i * dim + d], w * V[j * dim + d]);
    }

    __shared__ float ssum[256];
    ssum[tid] = local_sum;
    __syncthreads();
    for(int s = blockSize / 2; s > 0; s /= 2){
        if(tid < s)
            ssum[tid] += ssum[tid + s];
        __syncthreads();
    }

    if(tid == 0){
        m_out[i] = row_max;
        lout[i] = ssum[0];
    }
}

__global__
void merge_online_softmax(
    float *m_run, float *l_run, float *acc_run,
    const float *m_new, const float *l_new, const float *acc_new,
    int seq_q, int dim
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i >= seq_q)
        return;
    float m_old = m_run[i];
    float m_max = fmaxf(m_old, m_new[i]);

    float scale_old = expf(m_old - m_max);
    float scale_new = expf(m_new[i] - m_max);

    for(int d = 0; d < dim; d++){
        acc_run[i * dim + d] = scale_old * acc_run[i * dim + d] +
                                scale_new * acc_new[i * dim + d];
    }
    l_run[i] = scale_old * l_run[i] + scale_new * l_new[i];
    m_run[i] = m_max;
}

static void launch_attention_and_merge(
    int rank,
    cudaStream_t compute_stream,
    const float *d_Q, const float *kv,
    int seq_q, int seq_k, int dim,
    float *d_acc_local, float *d_m_local, float *d_l_local,
    float *m_run, float *l_run, float *acc_run,
    cudaEvent_t attn_start, cudaEvent_t attn_end, cudaEvent_t merge_end
) {
    CHECK_CUDA(cudaEventRecord(attn_start, compute_stream));
    gpu_attention_partial_kernel<<<seq_q, 256, 0, compute_stream>>>(
        d_Q, kv, kv,
        seq_q, seq_k, dim,
        d_acc_local, d_m_local, d_l_local
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(attn_end, compute_stream));

    merge_online_softmax<<<(seq_q + 255) / 256, 256, 0, compute_stream>>>(
        m_run, l_run, acc_run,
        d_m_local, d_l_local, d_acc_local,
        seq_q, dim
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(merge_end, compute_stream));
}

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(size < 2){
        if(rank == 0)
            printf("Need at least 2 processes. Got %d.\n", size);
        MPI_Finalize();
        return 1;
    }

    size_t kv_size_arg = 262144;
    int warmup = 10;
    int iters = 50;
    if(argc >= 2) kv_size_arg = (size_t)atol(argv[1]);
    if(argc >= 3) warmup = atoi(argv[2]);
    if(argc >= 4) iters = atoi(argv[3]);
    if(kv_size_arg == 0 || (kv_size_arg % (64 * sizeof(float))) != 0 ||
       warmup < 0 || iters <= 0){
        if(rank == 0)
            printf("Usage: %s <kv_size_bytes> [warmup>=0] [iters>0]\n"
                   "kv_size_bytes must be a positive multiple of %zu bytes.\n",
                   argv[0], 64 * sizeof(float));
        MPI_Finalize();
        return 1;
    }
    int local_rank = 0;
    int device = select_local_cuda_device(MPI_COMM_WORLD, rank, &local_rank);
    // GPU's information
    struct cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    printf("[Rank %d local_rank %d] GPU %d: %s (%.0f MB total)\n",
            rank, local_rank, device, prop.name, prop.totalGlobalMem / 1e6);
    
    // right neighbors
    int right = (rank + 1) % size;
    int left = (rank - 1 + size) % size;

    // dimensions
    int head_dim = 64;      //benchmark_single_gpu_shapes.py(head-dim)
    int seq_k = (int)(kv_size_arg / (head_dim * sizeof(float)));
    int seq_q = seq_k;
    int dim = head_dim;

    // KV shard size
    size_t kv_size = kv_size_arg;
    int n_floats = seq_k * dim;

    printf("[Rank %d] seq_q=%d, seq_k=%d, dim=%d, "
           "KV shard=%.1f KB\n",
           rank, seq_q, seq_k, dim, kv_size / 1024.0);

    // NCCL-initialization
    ncclUniqueId nccl_id;
    if(rank == 0){
        CHECK_NCCL(ncclGetUniqueId(&nccl_id));
    }
    // use MPI tell all rank
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t nccl_comm;
    CHECK_NCCL(ncclCommInitRank(&nccl_comm, size, nccl_id, rank));
    cudaStream_t comm_stream, compute_stream;
    CHECK_CUDA(cudaStreamCreate(&comm_stream));
    CHECK_CUDA(cudaStreamCreate(&compute_stream));
    cudaEvent_t comm_start_event, comm_end_event;
    cudaEvent_t attn_start_event, attn_end_event, merge_end_event;
    CHECK_CUDA(cudaEventCreate(&comm_start_event));
    CHECK_CUDA(cudaEventCreate(&comm_end_event));
    CHECK_CUDA(cudaEventCreate(&attn_start_event));
    CHECK_CUDA(cudaEventCreate(&attn_end_event));
    CHECK_CUDA(cudaEventCreate(&merge_end_event));


    // allocate two GPU buffers
    float *d_buf_1, *d_buf_2;
    CHECK_CUDA(cudaMalloc(&d_buf_1, kv_size));
    CHECK_CUDA(cudaMalloc(&d_buf_2, kv_size));
    
    float *h_init = (float *)malloc(kv_size);
    for(int i = 0; i < n_floats; i++)
        h_init[i] = rank * 1000.0f + (float)i;
    CHECK_CUDA(cudaMemcpy(d_buf_1, h_init, kv_size,
                          cudaMemcpyHostToDevice));

    // Q stays local on CPU
    float *Q = (float *)malloc(seq_q * dim * sizeof(float));
    for(int i = 0; i < seq_q * dim; i++)
        Q[i] = (rank * 500.0f + (float)(i % dim) * 0.01f) * 0.001f; // for ex

    float *m_run, *l_run, *acc_run;
    CHECK_CUDA(cudaMalloc(&m_run, seq_q * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&l_run, seq_q * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&acc_run, seq_q * dim * sizeof(float)));

    float *h_m_init = (float *)malloc(seq_q * sizeof(float));
    for(int i = 0; i < seq_q; i++)
        h_m_init[i] = -FLT_MAX;
    CHECK_CUDA(cudaMemcpy(m_run, h_m_init, seq_q * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(l_run, 0, seq_q * sizeof(float)));
    CHECK_CUDA(cudaMemset(acc_run, 0, seq_q * dim * sizeof(float)));

    /*
     * current_kv: the shard to SEND this step
     *             (starts as our own KV)
     * next_kv:    the buffer to RECEIVE into
     *
     * After each step we swap them:
     *   what we received becomes what we send next.
     */
    float *current_kv = d_buf_1;
    float *next_kv = d_buf_2;

    float *d_Q, *d_m_local;
    CHECK_CUDA(cudaMalloc(&d_Q, seq_q * dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_m_local, seq_q * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_Q, Q, seq_q * dim * sizeof(float),
                          cudaMemcpyHostToDevice));
    float *d_l_local;
    CHECK_CUDA(cudaMalloc(&d_l_local, seq_q * sizeof(float)));

    float *d_acc_local;
    CHECK_CUDA(cudaMalloc(&d_acc_local, seq_q * dim * sizeof(float)));   

    float *h_acc = (float *)malloc(seq_q * dim * sizeof(float));
    float *h_l = (float *)malloc(seq_q * sizeof(float));

    double sum_total = 0.0, sum_nccl = 0.0;
    double sum_attn = 0.0, sum_merge = 0.0, sum_final = 0.0;

    for(int iter = 0; iter < warmup + iters; iter++){
        CHECK_CUDA(cudaMemcpy(d_buf_1, h_init, kv_size,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(m_run, h_m_init, seq_q * sizeof(float),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(l_run, 0, seq_q * sizeof(float)));
        CHECK_CUDA(cudaMemset(acc_run, 0, seq_q * dim * sizeof(float)));
        current_kv = d_buf_1;
        next_kv = d_buf_2;

        double local_nccl = 0.0, local_attn = 0.0;
        double local_merge = 0.0, local_final = 0.0;

        MPI_Barrier(MPI_COMM_WORLD);
        double t_total_start = MPI_Wtime();

        for(int step = 0; step < size - 1; step++){
            CHECK_CUDA(cudaEventRecord(comm_start_event, comm_stream));
            CHECK_NCCL(ncclGroupStart());
            CHECK_NCCL(ncclSend(current_kv, n_floats, ncclFloat,
                                right, nccl_comm, comm_stream));
            CHECK_NCCL(ncclRecv(next_kv, n_floats, ncclFloat,
                                left, nccl_comm, comm_stream));
            CHECK_NCCL(ncclGroupEnd());
            CHECK_CUDA(cudaEventRecord(comm_end_event, comm_stream));

            launch_attention_and_merge(
                rank, compute_stream,
                d_Q, current_kv,
                seq_q, seq_k, dim,
                d_acc_local, d_m_local, d_l_local,
                m_run, l_run, acc_run,
                attn_start_event, attn_end_event, merge_end_event
            );

            CHECK_CUDA(cudaEventSynchronize(merge_end_event));
            CHECK_CUDA(cudaEventSynchronize(comm_end_event));

            float nccl_ms = 0.0f;
            float attn_ms = 0.0f;
            float merge_ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&nccl_ms, comm_start_event, comm_end_event));
            CHECK_CUDA(cudaEventElapsedTime(&attn_ms, attn_start_event, attn_end_event));
            CHECK_CUDA(cudaEventElapsedTime(&merge_ms, attn_end_event, merge_end_event));

            local_nccl += nccl_ms;
            local_attn += attn_ms;
            local_merge += merge_ms;

            float *tmp = current_kv;
            current_kv = next_kv;
            next_kv    = tmp;
        }

        launch_attention_and_merge(
            rank, compute_stream,
            d_Q, current_kv,
            seq_q, seq_k, dim,
            d_acc_local, d_m_local, d_l_local,
            m_run, l_run, acc_run,
            attn_start_event, attn_end_event, merge_end_event
        );
        CHECK_CUDA(cudaEventSynchronize(merge_end_event));
        float attn_ms = 0.0f;
        float merge_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&attn_ms, attn_start_event, attn_end_event));
        CHECK_CUDA(cudaEventElapsedTime(&merge_ms, attn_end_event, merge_end_event));
        local_attn += attn_ms;
        local_merge += merge_ms;

        double tf1 = MPI_Wtime();
        CHECK_CUDA(cudaMemcpy(h_acc, acc_run, seq_q * dim * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_l, l_run, seq_q * sizeof(float),
                              cudaMemcpyDeviceToHost));
        for(int i = 0; i < seq_q; i++)
            for(int d = 0; d < dim; d++)
                h_acc[i * dim + d] /= h_l[i];
        double tf2 = MPI_Wtime();
        local_final += (tf2 - tf1) * 1000.0;

        MPI_Barrier(MPI_COMM_WORLD);
        double local_total = (MPI_Wtime() - t_total_start) * 1000.0;
        double local_vals[5] = {
            local_total, local_nccl, local_attn, local_merge, local_final
        };
        double max_vals[5];
        MPI_Reduce(local_vals, max_vals, 5, MPI_DOUBLE,
                   MPI_MAX, 0, MPI_COMM_WORLD);

        if(rank == 0 && iter >= warmup){
            sum_total += max_vals[0];
            sum_nccl += max_vals[1];
            sum_attn += max_vals[2];
            sum_merge += max_vals[3];
            sum_final += max_vals[4];
        }
    }

    if(rank == 0){
        printf("nccl bench: warmup=%d iters=%d\n", warmup, iters);
        printf("avg_total=%.3f ms avg_NCCL=%.3f "
               "avg_attn=%.3f avg_merge=%.3f avg_final=%.3f\n",
               sum_total / iters, sum_nccl / iters,
               sum_attn / iters, sum_merge / iters, sum_final / iters);
    }

    char fname[64];
    sprintf(fname, "ring_output_rank%d.bin", rank);
    FILE *fp = fopen(fname, "wb");
    fwrite(h_acc, sizeof(float), seq_q * dim, fp);
    fclose(fp);
    printf("[Rank %d] output dumped to %s\n", rank, fname);

    // clean up
    free(Q);
    free(h_init);
    free(h_m_init);
    free(h_acc);
    free(h_l);
    CHECK_CUDA(cudaFree(m_run));
    CHECK_CUDA(cudaFree(l_run));
    CHECK_CUDA(cudaFree(acc_run));
    CHECK_CUDA(cudaFree(d_buf_1));
    CHECK_CUDA(cudaFree(d_buf_2));
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_m_local));
    CHECK_CUDA(cudaFree(d_l_local));
    CHECK_CUDA(cudaFree(d_acc_local));
    CHECK_CUDA(cudaEventDestroy(comm_start_event));
    CHECK_CUDA(cudaEventDestroy(comm_end_event));
    CHECK_CUDA(cudaEventDestroy(attn_start_event));
    CHECK_CUDA(cudaEventDestroy(attn_end_event));
    CHECK_CUDA(cudaEventDestroy(merge_end_event));
    CHECK_CUDA(cudaStreamDestroy(comm_stream));
    CHECK_CUDA(cudaStreamDestroy(compute_stream));
    CHECK_NCCL(ncclCommDestroy(nccl_comm));


    MPI_Finalize();
    return 0;

}

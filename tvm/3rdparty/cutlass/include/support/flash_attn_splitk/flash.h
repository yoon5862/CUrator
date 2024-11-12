/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include<iostream>
#include<cstdio>
#include<math.h>

#include<cuda.h>
#include<cutlass/cutlass.h>
#include <cutlass/numeric_types.h>


#include <cuda.h>
#include <vector>
#include "static_switch.h"
#include "kernel_traits.h"
#include "flash_fwd_kernel.h"

constexpr int TOTAL_DIM = 0;
constexpr int H_DIM = 1;
constexpr int D_DIM = 2;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params {
    using index_t = int64_t;
    // The QKV matrices.
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;

    // The number of heads.
    int h, h_k;
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads (query).
    int h_h_k_ratio; // precompute h / h_k,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_params : public Qkv_params {

    // The O matrix (output).
    void * __restrict__ o_ptr;
    void * __restrict__ oaccum_ptr;

    // The stride between rows of O.
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    // The pointer to the P matrix.
    void * __restrict__ p_ptr;

    // The pointer to the softmax sum.
    void * __restrict__ softmax_lse_ptr;
    void * __restrict__ softmax_lseaccum_ptr;

    // The dimensions.
    int b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim, total_q;

    // The scaling factors for the kernel.
    float scale_softmax;
    float scale_softmax_log2;

    // array of length b+1 holding starting offset of each sequence.
    int * __restrict__ cu_seqlens_q;
    int * __restrict__ cu_seqlens_k;
    int * __restrict__ leftpad_k;

    // If provided, the actual length of each k sequence.
    int * __restrict__ seqused_k;

    int *__restrict__ blockmask;

    // The K_new and V_new matrices.
    void * __restrict__ knew_ptr;
    void * __restrict__ vnew_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t knew_batch_stride;
    index_t vnew_batch_stride;
    index_t knew_row_stride;
    index_t vnew_row_stride;
    index_t knew_head_stride;
    index_t vnew_head_stride;

    // The cos and sin matrices for rotary embedding.
    void * __restrict__ rotary_cos_ptr;
    void * __restrict__ rotary_sin_ptr;

    // The indices to index into the KV cache.
    int * __restrict__ cache_batch_idx;

    // Paged KV cache
    int * __restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;

    // The dropout probability (probability of keeping an activation).
    float p_dropout;
    // uint32_t p_dropout_in_uint;
    // uint16_t p_dropout_in_uint16_t;
    uint8_t p_dropout_in_uint8_t;

    // Scale factor of 1 / (1 - p_dropout).
    float rp_dropout;
    float scale_softmax_rp_dropout;

    // Local window size
    int window_size_left, window_size_right;
    float softcap;


    // Pointer to the RNG seed (idx 0) and offset (idx 1).
    uint64_t * rng_state;

    bool is_bf16;
    bool is_causal;

    // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
    // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
    bool is_seqlens_k_cumulative;

    bool is_rotary_interleaved;

    int num_splits;  // For split-KV version

    void * __restrict__ alibi_slopes_ptr;
    index_t alibi_slopes_batch_stride;

    bool unpadded_lse;  // For varlen paths: LSE is in [nheads, total_seqlen_q] format instead of [b, nheads, seqlen_q].
    bool seqlenq_ngroups_swapped;  // q has been transposed from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d).
};


extern "C"{
    void set_params_fprop(Flash_fwd_params &params,
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      void *q,
                      void *k,
                      void *v,
                      void *out,
                      void *softmax_lse_d,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      const float softcap,
                      const size_t head_
                      ) {
  
    params = {};

    params.is_bf16 = false;
    params.q_ptr = q;
    params.k_ptr = k;
    params.v_ptr = v;

    params.q_row_stride = h * head_;
    params.k_row_stride = h_k * head_;
    params.v_row_stride = h_k * head_;
    params.q_head_stride = head_;
    params.k_head_stride = head_;
    params.v_head_stride = head_;
    params.o_ptr = out;
    params.o_row_stride = h * head_;
    params.o_head_stride = head_;

    params.q_batch_stride = seqlen_q * h * head_;
    params.k_batch_stride = seqlen_k * h * head_;
    params.v_batch_stride = seqlen_k * h * head_;
    params.o_batch_stride = seqlen_k * h * head_;

    params.cu_seqlens_q = nullptr;
    params.cu_seqlens_k = nullptr;
    params.seqused_k = nullptr;

    params.softmax_lse_ptr = softmax_lse_d;

    //set dimension
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    params.softcap = 0.0;
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;

    params.p_dropout = 0.f;
    params.rp_dropout = 1.f;
    params.scale_softmax_rp_dropout = 1.f;

    params.is_causal = true;

    params.window_size_left = seqlen_k;
    params.window_size_right = window_size_right;

    params.is_seqlens_k_cumulative = true;
    params.unpadded_lse = false;
    params.seqlenq_ngroups_swapped = false;

    params.alibi_slopes_ptr = nullptr;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_bwd_params : public Flash_fwd_params {

    // The dO and dQKV matrices.
    void *__restrict__ do_ptr;
    void *__restrict__ dq_ptr;
    void *__restrict__ dk_ptr;
    void *__restrict__ dv_ptr;

    // To accumulate dQ
    void *__restrict__ dq_accum_ptr;
    void *__restrict__ dk_accum_ptr;
    void *__restrict__ dv_accum_ptr;

    // // To accumulate dK and dV in case we're splitting the bwd along seqlen_q
    // dimension void *__restrict__ dk_accum_ptr; void *__restrict__
    // dv_accum_ptr;

    // The stride between rows of the dO, dQ, dK and dV matrices.
    // TD [2022-04-16]: We're using 32-bit indexing to save registers.
    // The code probably won't work for arrays larger than 2GB.
    index_t do_batch_stride;
    index_t do_row_stride;
    index_t do_head_stride;
    index_t dq_batch_stride;
    index_t dk_batch_stride;
    index_t dv_batch_stride;
    index_t dq_row_stride;
    index_t dk_row_stride;
    index_t dv_row_stride;
    index_t dq_head_stride;
    index_t dk_head_stride;
    index_t dv_head_stride;

    // The pointer to the softmax d sum.
    void *__restrict__ dsoftmax_sum;

    bool deterministic;
    index_t dq_accum_split_stride;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

#define ARCH_SUPPORTS_FLASH
#define KERNEL_PARAM_MODIFIER __grid_constant__

#define FLASH_UNSUPPORTED_ARCH printf("FATAL: FlashAttention requires building with sm version sm80-sm90, but was built for < 8.0!");

#define DEFINE_FLASH_FORWARD_KERNEL(kernelName, ...) \
template<typename Kernel_traits, __VA_ARGS__> \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Flash_fwd_params params)


// DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_kernel, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax) {
//     #if defined(ARCH_SUPPORTS_FLASH)
//         static_assert(!(Is_causal && Is_local)); // Enforce constraints
//         flash::compute_attn<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params);
//     #else
//         FLASH_UNSUPPORTED_ARCH
//     #endif
// }

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_splitkv_kernel, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV) {
    #if defined(ARCH_SUPPORTS_FLASH)
        flash::compute_attn_splitkv<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_splitkv_combine_kernel, int kBlockM, int Log_max_splits, bool Is_even_K) {
    static_assert(Log_max_splits >= 1);
    flash::combine_attn_seqk_parallel<Kernel_traits, kBlockM, Log_max_splits, Is_even_K>(params);
}



// template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
// void run_flash_fwd(Flash_fwd_params &params) {
//     constexpr size_t smem_size = Kernel_traits::kSmemSize;
//     // printf("smem_size = %d\n", smem_size);

//     // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
//     // https://github.com/kokkos/kokkos-kernels/issues/349
//     // https://github.com/HazyResearch/flash-attention/issues/21

//     const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
//     dim3 grid(num_m_block, params.b, params.h);
//     const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
//     const bool is_even_K = params.d == Kernel_traits::kHeadDim;
//     const bool return_softmax = params.p_ptr != nullptr;
//     BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
//         EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
//             LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
//                 BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
//                     ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
//                         SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
//                             auto kernel = &flash_fwd_kernel<Kernel_traits, Is_dropout && !Is_softcap, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, ReturnSoftmaxConst && Is_dropout && !Is_softcap>;
//                             if (smem_size >= 48 * 1024) cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
//                             kernel<<<grid, Kernel_traits::kNThreads, smem_size>>>(params);
//                         });
//                     });
//                 });
//             });
//         });
//     });
// }


template<typename Kernel_traits, bool Is_causal>
void run_flash_splitkv_fwd(Flash_fwd_params &params) {
    static_assert(!Kernel_traits::Is_Q_in_regs, "SplitKV implementation does not support Is_Q_in_regs");
    static_assert(!Kernel_traits::Share_Q_K_smem, "SplitKV implementation does not support Share_Q_K_smem");
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.num_splits > 1 ? params.num_splits : params.b, params.num_splits > 1 ? params.b * params.h : params.h);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
                BOOL_SWITCH(params.num_splits > 1, Split, [&] {
                    BOOL_SWITCH(params.knew_ptr != nullptr, Append_KV, [&] {
                        ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                            SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                                // If Append_KV, then we must have seqlen_offsets, which means cu_seqlens_k != nullptr.
                                // If not IsEvenKConst, we also set IsEvenMNConst to false to reduce number of templates.
                                // If Is_local, set Is_causal to false
                                auto kernel = &flash_fwd_splitkv_kernel<Kernel_traits, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && !Append_KV && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, Split, Append_KV>;
                                if (smem_size >= 48 * 1024) cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
                                kernel<<<grid, Kernel_traits::kNThreads, smem_size>>>(params);
                            });
                        });
                    });
                });
            });
        });
    });
    if (params.num_splits > 1) {
        // We want kBlockM to be as small as possible for more parallelism.
        // With 128 threads we can load 512 elements at a time, so if headdim is divisible by 128, kBlockM = 4.
        // If headdim is divisible by 64, then we set kBlockM = 8, etc.
        constexpr static int kBlockM = Kernel_traits::kHeadDim % 128 == 0 ? 4 : (Kernel_traits::kHeadDim % 64 == 0 ? 8 : 16);
        dim3 grid_combine((params.b * params.h * params.seqlen_q + kBlockM - 1) / kBlockM);
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            if (params.num_splits <= 2) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 1, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0>>>(params);
            } else if (params.num_splits <= 4) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 2, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0>>>(params);
            } else if (params.num_splits <= 8) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 3, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0>>>(params);
            } else if (params.num_splits <= 16) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 4, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0>>>(params);
            } else if (params.num_splits <= 32) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 5, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0>>>(params);
            } else if (params.num_splits <= 64) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 6, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0>>>(params);
            } else if (params.num_splits <= 128) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 7, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0>>>(params);
            }
        });
    }
}


///////////////////////////////////////////////////////////


// template<typename T, int Headdim_, bool Is_causal>
// void run_mha_fwd_hdim128_128(Flash_fwd_params &params) {
//     BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&]{
//         run_flash_fwd<Flash_fwd_kernel_traits<Headdim_, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params);
//     });
// }

// template<typename T, int Headdim_, bool Is_causal>
// void run_mha_fwd_hdim64_64(Flash_fwd_params &params) {
//     BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&]{
//         run_flash_fwd<Flash_fwd_kernel_traits<Headdim_, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params);
//     });
// }

template<typename T, int Headdim, bool Is_causal>
void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params) {
    constexpr static int kBlockM = 64;  // Fixed for all head dimensions
    // TD [2023-08-28]: nvcc segfaults for headdim 96 with block size 64 x 256,
    // and for headdim 192 with block size 64 x 128.
    // Also for headdim 160 with block size 64 x 128 after the rotary addition.
    constexpr static int kBlockN = Headdim <= 64 ? 256 : (Headdim <= 128 ? 128 : 64);
    run_flash_splitkv_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, 4, false, false, T>, Is_causal>(params);
}


////////////////////////////////////////////////////////////////////////////////////////////////////

// template<typename T, int Headdim, bool Is_causal>
// void run_mha_fwd_(Flash_fwd_params &params){
//     run_mha_fwd_hdim<T, Headdim, Is_causal>(params);
// }

// template<typename T, int Headdim, bool Is_causal> void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params, cudaStream_t stream);

// template<typename T, int Headdim, bool Is_causal> void run_mha_bwd_(Flash_bwd_params &params, cudaStream_t stream);


////////////////////////////////////////////////////////////////////////////////////////////////////


extern "C"{
// void run_mha_fwd128_128(Flash_fwd_params &params, bool force_split_kernel=false);

// void run_mha_fwd64_64(Flash_fwd_params &params, bool force_split_kernel=false);

void run_mha_fwd_splitk(Flash_fwd_params &params, bool force_split_kernel=true);

};
#pragma once




#define ARCH_SUPPORTS_FLASH
#define KERNEL_PARAM_MODIFIER __grid_constant__

#define FLASH_UNSUPPORTED_ARCH printf("FATAL: FlashAttention requires building with sm version sm80-sm90, but was built for < 8.0!");

#define DEFINE_FLASH_FORWARD_KERNEL(kernelName, ...) \
template<typename Kernel_traits, __VA_ARGS__> \
struct Flash_fwd_params; \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Flash_fwd_params params)


DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_kernel, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local)); // Enforce constraints
        // flash::compute_attn<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    // printf("smem_size = %d\n", smem_size);

    // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
    // https://github.com/kokkos/kokkos-kernels/issues/349
    // https://github.com/HazyResearch/flash-attention/issues/21

    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
                BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
                    ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                        SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                            // auto kernel = &flash_fwd_kernel<Kernel_traits, Is_dropout && !Is_softcap, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, ReturnSoftmaxConst && Is_dropout && !Is_softcap>;

                            // if (smem_size >= 48 * 1024) {
                            //     cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
                            // }
                            // int ctas_per_sm;
                            // cudaError status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                            //     &ctas_per_sm, kernel, Kernel_traits::kNThreads, smem_size);
                            // printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);
                            // kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                        });
                    });
                });
            });
        });
    });
}



template<typename T, int Headdim_, bool Is_causal>
void run_mha_fwd_hdim(Flash_fwd_params &params) {
    constexpr static int Headdim = Headdim_;
    
    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&]{
      run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params);
    });
}
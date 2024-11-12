/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Template for a pipelined GEMM kernel. Does not compute batching or support split-K.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"
#include "cutlass/arch/arch.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Mma_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Epilogue_,             ///! Epilogue
  typename ThreadblockSwizzle_,   ///! Threadblock swizzling function
  bool SplitKSerial               ///! If true, code supporting split-K via serial reduction is enabled.
>
struct GemmBatched {

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using OutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static bool const kSplitKSerial = SplitKSerial;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  /// Parameters structure
  struct Params {
    cutlass::gemm::GemmCoord problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    int swizzle_log_tile;
    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorA::TensorRef ref_A;
    int64_t stride_A;
    typename Mma::IteratorB::Params params_B;
    typename Mma::IteratorB::TensorRef ref_B;
    int64_t stride_B;
    typename Epilogue::OutputTileIterator::Params params_C;
    typename Epilogue::OutputTileIterator::TensorRef ref_C;
    int64_t stride_C;
    typename Epilogue::OutputTileIterator::Params params_D;
    typename Epilogue::OutputTileIterator::TensorRef ref_D;
    int64_t stride_D;
    typename OutputOp::Params epilogue;
    int batch_count;
    int gemm_k_iterations;

    //for k-split
    int *semaphore;
    int gemm_k_size;
    int k_split_slices;
    

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params() : swizzle_log_tile(0) { }

    CUTLASS_HOST_DEVICE
    Params(
      cutlass::gemm::GemmCoord const & problem_size_,
      cutlass::gemm::GemmCoord const & grid_tiled_shape_,
      typename Mma::IteratorA::TensorRef ref_A_,
      int64_t stride_A_,
      typename Mma::IteratorB::TensorRef ref_B_,
      int64_t stride_B_,
      typename Epilogue::OutputTileIterator::TensorRef ref_C_,
      int64_t stride_C_,
      typename Epilogue::OutputTileIterator::TensorRef ref_D_,
      int64_t stride_D_,
      typename OutputOp::Params epilogue_,
      int batch_count_,
      int *workspace = nullptr,
      int k_split_slices = 1
    ):
      problem_size(problem_size_),
      grid_tiled_shape(grid_tiled_shape_),
      swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape)),
      params_A(ref_A_.layout()),
      ref_A(ref_A_),
      stride_A(stride_A_),
      params_B(ref_B_.layout()),
      ref_B(ref_B_),
      stride_B(stride_B_),
      params_C(ref_C_.layout()),
      ref_C(ref_C_),
      stride_C(stride_C_),
      params_D(ref_D_.layout()),
      ref_D(ref_D_),
      stride_D(stride_D_),
      epilogue(epilogue_),
      batch_count(batch_count_),
      k_split_slices(k_split_slices){
        int total_gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;
        gemm_k_iterations = (total_gemm_k_iterations + k_split_slices - 1) / k_split_slices;

        gemm_k_size = gemm_k_iterations * Mma::Shape::kK;
        semaphore = workspace;
    }
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  GemmBatched() { } 

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    int real_m = params.grid_tiled_shape.m() / params.batch_count;
    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() % real_m ||
      params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {

      return;
    }

    int batch_idx = threadblock_tile_offset.m() / real_m;
    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A{
      (threadblock_tile_offset.m() % real_m) * Mma::Shape::kM,
      threadblock_tile_offset.k() * params.gemm_k_size,
    };

    cutlass::MatrixCoord tb_offset_B{
      threadblock_tile_offset.k() * params.gemm_k_size,
      threadblock_tile_offset.n() * Mma::Shape::kN
    };

    int problem_size_k = min(params.problem_size.k(), (threadblock_tile_offset.k() + 1) * params.gemm_k_size);
    int gemm_k_iterations = (problem_size_k - tb_offset_A.column() + Mma::Shape::kK - 1) / Mma::Shape::kK;
    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Construct iterators to A and B operands
    typename Mma::IteratorA iterator_A(
      params.params_A,
      params.ref_A.data(),
      {params.problem_size.m(), problem_size_k},
      thread_idx,
      tb_offset_A);

    iterator_A.add_pointer_offset(params.stride_A * batch_idx);

    typename Mma::IteratorB iterator_B(
      params.params_B,
      params.ref_B.data(),
      {problem_size_k, params.problem_size.n()},
      thread_idx,
      tb_offset_B);

    iterator_B.add_pointer_offset(params.stride_B * batch_idx);


    //
    // Main loop
    //

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = canonical_warp_idx();

    int lane_idx = threadIdx.x % 32;
    
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename Mma::FragmentC accumulators;

    accumulators.clear();


   if(!kSplitKSerial || gemm_k_iterations > 0){
     // Compute threadblock-scoped matrix multiply-add
    mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);
   }

    //
    // Epilogue
    //

    OutputOp output_op(params.epilogue);

    //
    // Masked tile iterators constructed from members
    //

    threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    //assume identity swizzle
    MatrixCoord threadblock_offset(
      (threadblock_tile_offset.m() % real_m) * Mma::Shape::kM,
      threadblock_tile_offset.n() * Mma::Shape::kN
    );

    int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

    Semaphore semaphore(params.semaphore + block_idx, thread_idx);

    if(kSplitKSerial && params.grid_tiled_shape.k() > 1){
      semaphore.fetch();
      output_op.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
    }

    // Tile iterator writing to output tile
    typename Epilogue::OutputTileIterator iterator_C(
      params.params_C,
      params.ref_C.data(),
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset
    );

    iterator_C.add_pointer_offset(params.stride_C * batch_idx);

    // Tile iterator writing to output tile
    typename Epilogue::OutputTileIterator iterator_D(
      params.params_D,
      params.ref_D.data(),
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset
    );

    iterator_D.add_pointer_offset(params.stride_D * batch_idx);

    Epilogue epilogue(
      shared_storage.epilogue, 
      thread_idx, 
      warp_idx, 
      lane_idx);
    
    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
        
      // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
      if (threadblock_tile_offset.k()) {
        iterator_C = iterator_D;
      }

      semaphore.wait(threadblock_tile_offset.k());

    }

    // run efficient epilogue
    epilogue(output_op, iterator_D, accumulators, iterator_C);

    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
      
      int lock = 0;
      if (params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1) {

        // The final threadblock resets the semaphore for subsequent grids.
        lock = 0;
      }
      else {
        // Otherwise, the semaphore is incremented
        lock = threadblock_tile_offset.k() + 1;
      }

      semaphore.release(lock);
    }

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass
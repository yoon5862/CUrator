// This header file is for cutlass support
// cutlass softmax use extra gpu malloc for softmax kerenel

#pragma once

#include <cub/cub.cuh>
#include <math_constants.h>
#include <assert.h>
#include <cuda.h>
#include<cuda_runtime.h>
#include<stdio.h>
#include<math.h>


namespace cutlass{
  namespace support{
    template<typename T>
      struct SumOp{
          __device__ __forceinline__ T operator()(const T& a, const T& b) const{return a + b;}
      };

      template<typename T>
      struct MaxOp{
          __device__ __forceinline__ T operator()(const T& a, const T& b) const{return max(a, b);}
      };

      template<typename ReductionOp, typename T, int thread_group_width=32>
      __inline__ __device__ T WarpAllReduce(T val){
        ReductionOp operation;
          for(int mask=thread_group_width / 2; mask > 0; mask /= 2){
              val = operation(val,  __shfl_xor_sync(0xffffffff, val, mask));
          }
          return val;
      }

      template<typename T>
      __inline__ __device__ T Inf();

      template<>
      __inline__ __device__ float Inf<float>() {
        return CUDART_INF_F;
      }

      template<>
      __inline__ __device__ double Inf<double>() {
        return CUDART_INF;
      }

      template<typename T>
      __inline__ __device__ T Exp(T x);

      template<>
      __inline__ __device__ float Exp<float>(float x) {
        return expf(x);
      }

      template<typename T>
      __inline__ __device__ T Div(T a, T b);

      template<>
      __inline__ __device__ float Div<float>(float a, float b) {
        return a / b;
      }

      template<>
      __inline__ __device__ double Div<double>(double a, double b) {
        return a / b;
      }


      template<typename ElementD,
              int _row_per_access,
              int _pack_size,
              int _col_per_thread
      >
      __global__ void softmaxKernel(const ElementD *input, ElementD *output,
                                    const int64_t rows, const int64_t columns){
        const int row_per_access = _row_per_access;
        const int col_per_thread = _col_per_thread;
        const int pack_size = _pack_size;
        const int num_packs = col_per_thread / pack_size;

        static_assert(col_per_thread % pack_size == 0, "alignment sould be devided by register file size");
        assert(columns <= col_per_thread * 32);

        ElementD buf[row_per_access][col_per_thread]; //store data in register

        const int global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y;
        const int num_global_thread_group = gridDim.x * blockDim.y;
        const int land_id = threadIdx.x; // lane id

        const int64_t step = num_global_thread_group * row_per_access;
        const int64_t batch_stride = blockIdx.y * (rows * columns);

        
        for(int64_t row = global_thread_group_id * row_per_access; row < rows; row += step){
          ElementD thread_max[row_per_access];

      #pragma unroll
          for(int row_id = 0; row_id < row_per_access; row_id++){
            thread_max[row_id] = -Inf<ElementD>();
            ElementD *row_buf = buf[row_id];

      #pragma unroll
            for(int pack_id = 0; pack_id < num_packs; pack_id++){
              const int col = (pack_id * 32 + land_id) * pack_size;

              //load data
              // if(row+row_id >= rows) return;
      #pragma unroll
            for(int i = 0; i < pack_size; i++){
              if(col + i >= columns) row_buf[i] = -Inf<ElementD>();
              row_buf[pack_id * pack_size + i] = input[batch_stride + (row+row_id) * columns + col + i];
              // printf("%d, %d: %f\n", threadIdx.x, i+pack_id, input[row * columns + col + i]);
            }
              
      #pragma unroll
            for(int i = 0; i < pack_size; i++){
              thread_max[row_id] = max(thread_max[row_id], row_buf[pack_id * pack_size + i]); //here is error!
            }
            }
          }

          ElementD warp_max[row_per_access];
      #pragma unroll
          for(int row_id = 0; row_id < row_per_access; row_id++) warp_max[row_id] = WarpAllReduce<MaxOp<ElementD>, ElementD, 32>(thread_max[row_id]);

          ElementD thread_sum[row_per_access];
      #pragma unroll
          for(int row_id = 0; row_id < row_per_access; row_id++){
            thread_sum[row_id] = 0;
            ElementD *row_buf = buf[row_id];
      #pragma unroll
            for(int i = 0; i < col_per_thread; i++){
              row_buf[i] = Exp(row_buf[i] - warp_max[row_id]);
              thread_sum[row_id] += row_buf[i];
            }
          }

          ElementD warp_sum[row_per_access];
        #pragma unroll
          for(int row_id = 0; row_id < row_per_access; row_id++) warp_sum[row_id] = WarpAllReduce<SumOp<ElementD>, ElementD, 32>(thread_sum[row_id]);

      #pragma unroll
          for(int row_id = 0; row_id < row_per_access; row_id++){
            ElementD *row_buf = buf[row_id];
      #pragma unroll
            for(int i = 0; i < col_per_thread; i++)row_buf[i] = row_buf[i] / warp_sum[row_id];

      #pragma unroll
            for(int i = 0; i < num_packs; i++){
              const int col = (i * 32 + land_id) * pack_size;
              //sotre data
              if(row+row_id >= rows) return;
      #pragma unroll
              for(int j = 0; j < pack_size; j++){
                if(col + j >= columns) return;
                output[batch_stride + (row + row_id) * columns + col + j] = row_buf[i * pack_size + j];
              }
            }
          }
        }
      }

      template<
              int _pack_size,
              int _cols_per_thread,
              int _warp_count
      >
      bool can_implement(const int64_t batch, const int64_t rows, const int64_t cols){
        const int pack_size = _pack_size;
        const int cols_per_thread = _cols_per_thread;
        
        if(cols_per_thread % pack_size != 0) return false;
        if(cols_per_thread * 32 != cols) return false;

        return true;
      }

      //softmax kernel for cutlass support
      //https://oneflow2020.medium.com/how-to-implement-an-efficient-softmax-cuda-kernel-oneflow-performance-optimization-sharing-405ad56e9031
      template<typename ElementD,
              int _row_per_access,
              int _pack_size,
              int _cols_per_thread,
              int _warp_count
      >
      void softmaxWarp(ElementD *load, ElementD *store, const int64_t batch, const int64_t rows, const int64_t cols){
        constexpr int Kwarp = 32; // warp size
        constexpr int Kwarp_count = _warp_count; //warpPerCTA
        constexpr int row_per_access = _row_per_access; //rowPerWarp

        constexpr int col_per_thread = _cols_per_thread; //dataPerThread
        constexpr int pack_size = _pack_size; //vectorization shape, 128bit is best performance

        //calculate for Grid Dimension
        const int block_x = ((rows + row_per_access - 1)/row_per_access + Kwarp_count - 1) / Kwarp_count;
        const int block_y = batch; //y dimension for cutlass

        dim3 blockDimension(block_x, block_y);
        dim3 threadDimension(Kwarp, Kwarp_count);

        softmaxKernel<ElementD, row_per_access, pack_size, col_per_thread><<<blockDimension, threadDimension>>>(load, store, rows, cols);
      }
    
  }
}
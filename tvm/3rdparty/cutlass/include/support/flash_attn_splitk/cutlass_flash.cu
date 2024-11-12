#include<iostream>
#include <unistd.h>
#include<string>
#include<fstream>
#include "cutlass/cutlass.h"
#include "kernel_forward.h"
using namespace std;

#ifndef Q_TILE
#define Q_TILE 64
#endif

#ifndef K_TILE
#define K_TILE 64
#endif

#ifndef KMAX_TILE
#define KMAX_TILE 64
#endif

int main(int argc, char *argv[]){
  int batch = 1;
  int seq_len = 512;
  int head_num = 12;
  int head_size = 64;
  std::string rlt_json_dir = "";

  int option;
  while((option = getopt(argc, argv, "b:s:n:h:d:")) != -1){
      switch(option){
          case 'b':
              batch = std::stoi(optarg);
              break;
          case 's':
              seq_len = std::stoi(optarg);
              break;
          case 'n':
              head_num = std::stoi(optarg);
              break;
            case 'h':
              head_size = std::stoi(optarg);
              break;
            case 'd':
            rlt_json_dir = std::string(optarg);
            break;
          case '?':
              break;
      }
  }

  // std::cout << batch << " " << seq_len << " " << head_num << " " << head_size << std::endl;

  cutlass::half_t *q, *k, *v, *o;

  const size_t data_size = batch * seq_len * head_num * head_size;

  cudaMalloc(&q, sizeof(cutlass::half_t) * data_size);
  cudaMalloc(&k, sizeof(cutlass::half_t) * data_size);
  cudaMalloc(&v, sizeof(cutlass::half_t) * data_size);
  cudaMalloc(&o, sizeof(cutlass::half_t) * data_size);

  cudaMemset(q, 1, sizeof(cutlass::half_t) * data_size);
  cudaMemset(k, 1, sizeof(cutlass::half_t) * data_size);
  cudaMemset(v, 1, sizeof(cutlass::half_t) * data_size);
  cudaMemset(o, 1, sizeof(cutlass::half_t) * data_size);

  
  static constexpr int kMaxK = KMAX_TILE;
  static int const kQueriesPerBlock = Q_TILE;
  static int const kKeysPerBlock = K_TILE;

  using Attention = AttentionKernel<
  cutlass::half_t,
  cutlass::arch::Sm80,
  true,
  kQueriesPerBlock,
  kKeysPerBlock,
  kMaxK,
  false,
  false
  >;


  typename Attention::Params p;
  {
    p.query_ptr = q;
    p.key_ptr = k;
    p.value_ptr = v;
    p.logsumexp_ptr = nullptr;
    p.output_accum_ptr = nullptr;

    if(Attention::kNeedsOutputAccumulatorBuffer){
      cudaMalloc(&p.output_accum_ptr, sizeof(cutlass::half_t) * data_size);
    }
    p.output_ptr = o;
    p.scale = 1.0f / sqrt(float(head_size));

    p.num_heads = head_num;
    p.num_batches = batch;
    p.head_dim = head_size;
    p.head_dim_value = head_size;
    p.num_queries = seq_len;
    p.num_keys = seq_len;
    p.custom_mask_type = Attention::CausalFromTopLeft;

    p.q_strideH = head_size;
    p.k_strideH = head_size;
    p.v_strideH = head_size;

    p.q_strideM = int32_t(head_num * head_size);
    p.k_strideM = int32_t(head_num * head_size);
    p.v_strideM = int32_t(head_num * head_size);

    p.q_strideB = p.q_strideM * seq_len;
    p.k_strideB = p.k_strideM * seq_len;
    p.v_strideB = p.v_strideM * seq_len;
  }


  constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
  int smem_bytes = sizeof(typename Attention::SharedStorage);

  cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

  if(!Attention::check_supported(p)){
    std::cerr << "Kernel does not support these inputs" << std::endl;
    return 0;
  }

  for(int i = 0; i < 10; i++) kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);

  cudaEvent_t start, end;
  float elasped_time = 0.0f;

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start);
  for(int i = 0; i < 10; i++) kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);
  cudaEventRecord(end);
  cudaEventSynchronize(end);

  cudaEventElapsedTime(&elasped_time, start, end);
  
  std::fstream dataFile;
  std::string fileName = rlt_json_dir + "/"  + std::to_string(batch) + "_" + std::to_string(seq_len) + "_" + std::to_string(head_num) + "_" + std::to_string(head_size) + ".json";
  std::string json = "{\"dim\": [" + std::to_string(kQueriesPerBlock) + ", " + std::to_string(kKeysPerBlock) + ", " + std::to_string(kMaxK) +  "], " + "\"time\": " + std::to_string(elasped_time / 10) +  "}";

  dataFile.open(fileName, std::ios::app);
  dataFile << json << std::endl;

  // std::cout << json << std::endl;

  return 0;
}
#include<iostream>
#include<vector>
#include<cmath>
#include <cstdio>
#include <unistd.h>
#include<fstream>

#include<cuda.h>

#include<cutlass/cutlass.h>
#include <cutlass/numeric_types.h>



#include "flash.h"
#include "static_switch.h"

using namespace std;



int main(int argc, char *argv[]){

  

  int batch_size = 1; /*batch*/
  int seq_len = 512; /*seq_len*/
  int num_heads = 12; /*num_heads*/
  int head_size = 64; /*head_size*/
  int split_k = 1;

  const int window_size_left = -1, window_size_right = 0;
  const float softcap = 0.0f;
  const float softmax_scale = std::sqrt(head_size);

  int option;
  while((option = getopt(argc, argv, "b:s:n:h:s:")) != -1){
      switch(option){
          case 'b':
              batch_size = std::stoi(optarg);
              break;
          case 'n':
              num_heads = std::stoi(optarg);
              break;
          case 'h':
              head_size = std::stoi(optarg);
              break;
          case 's':
              split_k = std::stoi(optarg);
              break;
          case '?':
              break;
      }
  }

  cutlass::half_t *q, *k, *v, *out;
  float *softmax_lse;

  cudaMalloc(&q, (batch_size * seq_len * num_heads * head_size) * sizeof(cutlass::half_t));
  cudaMalloc(&k, (batch_size * seq_len * num_heads * head_size) * sizeof(cutlass::half_t));
  cudaMalloc(&v, (batch_size * seq_len * num_heads * head_size) * sizeof(cutlass::half_t));
  cudaMalloc(&out, (batch_size * seq_len * num_heads * head_size) * sizeof(cutlass::half_t));


  cudaMemset(q, 1, (batch_size * seq_len * num_heads * head_size) * sizeof(cutlass::half_t));
  cudaMemset(k, 1, (batch_size * seq_len * num_heads * head_size) * sizeof(cutlass::half_t));
  cudaMemset(v, 1, (batch_size * seq_len * num_heads * head_size) * sizeof(cutlass::half_t));
  cudaMemset(out, 0, (batch_size * seq_len * num_heads * head_size) * sizeof(cutlass::half_t));


  cutlass::half_t *q_padded = q;
  cutlass::half_t *k_padded = k;
  cutlass::half_t *v_padded = v;

  auto round_multiple = [](int x, int m) {return (x + m - 1) / m * m; };
  const int head_size_og = round_multiple(head_size, 8);
  const int head_size_rounded = head_size <= 192 ? round_multiple(head_size, 32) : 256;
  const int seq_len_rounded = round_multiple(seq_len, 128);


  cudaMalloc(&softmax_lse, (batch_size * head_size_og * seq_len) * sizeof(float));
  cudaMemset(softmax_lse, 0, (batch_size * head_size_og * seq_len) * sizeof(float)); // need to be allocated in build time

  for(int i = 0; i < 100; i++) cudaMemset(softmax_lse, 0, (batch_size * head_size_og * seq_len) * sizeof(float)); // need to be allocated in build time

  cudaEvent_t start, end;

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start);

  Flash_fwd_params params;
  set_params_fprop(params,
                   batch_size,
                   seq_len, seq_len,
                   seq_len_rounded, seq_len_rounded,
                   num_heads, num_heads,
                   head_size_og, head_size_rounded,
                   q_padded, k_padded, v_padded, out,
                   softmax_lse,
                   softmax_scale,
                   window_size_left,
                   window_size_right,
                   softcap, head_size);
  
  params.num_splits = split_k;

//   split-k buffer in build time
//   softmax_lse_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
//   out_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q, head_size_rounded}, opts.dtype(at::kFloat));
//   params.num_splits
//   params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
//   params.oaccum_ptr = out_accum.data_ptr();

  run_mha_fwd_splitk(params);


  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float elasped_time;
  cudaEventElapsedTime(&elasped_time, start, end);

  // std::cout << elasped_time << std::endl;

  std::fstream dataFile;
  std::string fileName = "./flash_atten_rlt/" + std::to_string(batch_size) + "_" + std::to_string(seq_len) + "_" + std::to_string(num_heads)
                         + "_" + std::to_string(head_size) + ".json";
  std::string json = "{\"split_k\": " + std::to_string(split_k) + ", \"time\": " + std::to_string(elasped_time) + "}";

  dataFile.open(fileName, std::ios::app);
  dataFile << json << std::endl;

  // std::cout << fileName << std::endl;
  // std::cout << json << std::endl;

  return 0;
}
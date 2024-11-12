#include<iostream>
#include<cuda_runtime.h>
#include"./softmax.cuh"
using namespace std;

int main(void){
  const int batch = 2;
  const int M = 512;
  const int N = 512;

  float *input = new float[batch * M * N];
  float *output = new float[batch * M * N];

  //input matrix
  for(int i = 0; i < batch * M * N; i++) input[i] = float(i) * 0.001;
  for(int i = 0; i < batch * M * N; i++) output[i] = 0.0f;


  float *d_in, *d_out;
  cudaMalloc(&d_in, batch * M * N * sizeof(float));
  cudaMalloc(&d_out, batch * M * N * sizeof(float));

  cudaMemcpy(d_in, input, batch * M * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, output, batch * M * N * sizeof(float), cudaMemcpyHostToDevice);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaDeviceSynchronize();

  bool can_implement = cutlass::support::can_implement<2, 16, 4>(batch, M, N);
  if(can_implement == false){
    return 0;
  }
  
  //warmming up gpu
  for(int i = 0; i < 100; i++) cutlass::support::softmaxWarp<float, 1, 2, 16, 4>(d_in, d_out, batch, M, N);

  cudaEventRecord(start);
  cutlass::support::softmaxWarp<float, 1, 2, 16, 4>(d_in, d_out, batch, M, N);
  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float time;
  cudaEventElapsedTime(&time, start, end);
  std::cout << time << std::endl;

  return 0;
}
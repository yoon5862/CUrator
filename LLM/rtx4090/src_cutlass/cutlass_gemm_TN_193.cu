
#include<iostream>
#include<cuda_runtime.h>

#include <unistd.h>
#include<string>
#include<fstream>      

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"
#include <cutlass/util/host_tensor.h>


float cutlass_strided_bathed_sgemm_0(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::Gemm<
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::ColumnMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm86,
                                                    cutlass::gemm::GemmShape<96, 256, 8>,
                                                    cutlass::gemm::GemmShape<48, 128, 8>,
                                                    cutlass::gemm::GemmShape<1, 1, 1>,
                                                    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
                                                    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
                                                    2,
                                                    1,
                                                    1,
                                                    true,
                                                    cutlass::arch::OpMultiplyAdd
                                                    >;
    
    Gemm gemm_op;
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    Gemm::Arguments arguments{
            {m, n, k},
            {A, lda},
            {B, ldb},
            {C, ldc},
            {C, ldc},
            {alpha, beta},
            split_k
        };
        
        int *workspace;
        int workspace_size = ((m + (96 - 1)) / 96) * ((m + (256 - 1)) / 256) * sizeof(int);
        cudaMalloc(&workspace, workspace_size);
        cudaMemset(workspace, 0, workspace_size);
        
        cutlass::Status status = gemm_op.can_implement(arguments);
        if(status != cutlass::Status::kSuccess) return -1;
        
        status = gemm_op.initialize(arguments, workspace);
        if(status != cutlass::Status::kSuccess) return -1;
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
        status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "./rtx4090/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[96, 256, 8], [48, 128, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + ", \"sizzle\": 8" + " , \"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }
            cudaFree(workspace);
            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);
    
    cudaFree(workspace);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "./rtx4090/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[96, 256, 8], [48, 128, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + ", \"sizzle\": 8" + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        



int main(int argc, char *argv[]){
    float *rlt = new float[1];

    int M = 64;
    int N = 64;
    int K = 64;
    int Batch = 1;
    int split_k = 1;
    int device_id;
    
    int option;
    while((option = getopt(argc, argv, "m:n:k:b:s:d:")) != -1){
        switch(option){
            case 'm':
                M = std::stoi(optarg);
                break;
            case 'n':
                N = std::stoi(optarg);
                break;
            case 'k':
                K = std::stoi(optarg);
                break;
            case 'b':
                Batch = std::stoi(optarg);
                break;
            case 's':
                split_k = std::stoi(optarg);
                break;
            case 'd':
                device_id = std::stoi(optarg);
                break;    
            case '?':
                break;
        }
    }
    
    cudaSetDevice(device_id);
    
    int const lda = K;
    int const ldb = K;
    int const ldc = N;
    
    int const count_A = Batch * M * K;
    int const count_B = Batch * N * K;
    int const count_C = Batch * M * N;
    
    long long int batch_stride_A = static_cast<long long int>(M) * static_cast<long long int>(K);
    long long int batch_stride_B = static_cast<long long int>(K) * static_cast<long long int>(N);
    long long int batch_stride_C = static_cast<long long int>(M) * static_cast<long long int>(N);
    
    float alpha = static_cast<float>(1.0f);
    float beta = static_cast<float>(0.0f);
    
    std::vector<float> host_A(count_A, 1.2f);
    std::vector<float> host_B(count_B, 1.0f);
    std::vector<float> host_C(count_C);
    
    float *A;
    float *B;
    float *C;
    
    cudaMalloc(&A, count_A * sizeof(float));
    cudaMalloc(&B, count_B * sizeof(float));
    cudaMalloc(&C, count_C * sizeof(float));
    
    cudaMemcpy(A, host_A.data(), count_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B, host_B.data(), count_B * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C, host_C.data(), count_C * sizeof(float), cudaMemcpyHostToDevice);
    
    //warmp up
    for(int i = 0; i < 2; i++){
        cutlass_strided_bathed_sgemm_0(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, 1, 1);
    }
    
    rlt[0] = cutlass_strided_bathed_sgemm_0(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	
    
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    
    return 0;
}
        
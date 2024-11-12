#!/bin/bash
#SBATCH -J my_job_name
#SBATCH --gres=gpu:1
#SBATCH --partition=suma_rtx4090
#SBATCH --output=3090_%j.txt
#SBATCH --time=2-00:00:00


export PATH="/home/5862www/cuda-tool/cuda-12.0/bin:$PATH"
export LD_LIBRARY_PATH="/home/5862www/cuda-tool/cuda-12.0/lib64:$LD_LIBRARY_PATH"

export CC=/home/5862www/gcc/bin/gcc
export CXX=/home/5862www/gcc/bin/g++

export PYTHONPATH=/home/5862www/curator/tvm/python

# nvcc flash.cu -arch=sm_86 --std=c++17 -I/home/5862www/curator/tvm/3rdparty/cutlass/include/support/cutlass-3.5.1/include -o flash --expt-relaxed-constexpr --expt-extended-lambda -O3 -t 8 -use_fast_math 

# nvcc flash_attn.cu -arch=sm_80 --std=c++17 -Xcompiler -fPIC -shared -I/home/5862www/curator/tvm/3rdparty/cutlass/include/support/cutlass-3.5.1/include -o flash_attn_splitk.so --expt-relaxed-constexpr --expt-extended-lambda -O3 -use_fast_math

# nvcc -O3 main.cu -arch=sm_86 --std=c++17 -I/home/5862www/curator/tvm/3rdparty/cutlass/include/support/cutlass-3.5.1/include -L. flash_attn64_64.so -o flash_attn_test

nvcc flash.cu -arch=sm_80 --std=c++17 -I/home/5862www/curator/tvm/3rdparty/cutlass/include/support/cutlass-3.5.1/include -o test.o --expt-relaxed-constexpr --expt-extended-lambda -O3 -use_fast_math -L. flash_attn_splitk.so
#!/bin/bash


sm=$1
tmp_dir=$2

cd ../LLM

echo "Our Evaluation is Reported in curator/LLM/$tmp_dir"

#CUTLASS Batch = 2
python figure_10.py  --batch=2 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir

#CUTLASS Batch = 4
python figure_10.py  --batch=4 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir

#CUTLASS Batch = 8
python figure_10.py  --batch=8 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir

#CUTLASS Batch = 12
python figure_10.py  --batch=12 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir

#CUTLASS Batch = 16
python figure_10.py  --batch=16 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir


#cuBLAS Batch = 2
python figure_10.py  --batch=2 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir

#cuBLAS Batch = 4
python figure_10.py  --batch=4 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir

#cuBLAS Batch = 8
python figure_10.py  --batch=8 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir

#cuBLAS Batch = 12
python figure_10.py  --batch=12 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir

#cuBLAS Batch = 16
python figure_10.py  --batch=16 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir


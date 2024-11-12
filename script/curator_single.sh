#!/bin/bash


sm=$1
tmp_dir=$2

cd ../LLM

echo "Our Evaluation is Reported in curator/LLM/$tmp_dir text file"

#CUTLASS BERT tiny
python curator.py --model=gaunernst/bert-tiny-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --tmp_dir=$tmp_dir
python curator.py --model=gaunernst/bert-tiny-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --tmp_dir=$tmp_dir
python curator.py --model=gaunernst/bert-tiny-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --tmp_dir=$tmp_dir


# CUTLASS BERT mini
python curator.py --model=gaunernst/bert-mini-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --tmp_dir=$tmp_dir
python curator.py --model=gaunernst/bert-mini-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --tmp_dir=$tmp_dir
python curator.py --model=gaunernst/bert-mini-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --tmp_dir=$tmp_dir

# CUTLASS BERT small
python curator.py --model=gaunernst/bert-small-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --tmp_dir=$tmp_dir
python curator.py --model=gaunernst/bert-small-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --tmp_dir=$tmp_dir
python curator.py --model=gaunernst/bert-small-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --tmp_dir=$tmp_dir

# CUTLASS BERT medium
python curator.py --model=gaunernst/bert-medium-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --tmp_dir=$tmp_dir
python curator.py --model=gaunernst/bert-medium-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --tmp_dir=$tmp_dir
python curator.py --model=gaunernst/bert-medium-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --tmp_dir=$tmp_dir

# CUTLASS BERT base
python curator.py --model=bert-base-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --tmp_dir=$tmp_dir
python curator.py --model=bert-base-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --tmp_dir=$tmp_dir
python curator.py --model=bert-base-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --tmp_dir=$tmp_dir

# CUTLASS BERT large
python curator.py --model=bert-large-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --tmp_dir=$tmp_dir
python curator.py --model=bert-large-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --tmp_dir=$tmp_dir
python curator.py --model=bert-large-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --tmp_dir=$tmp_dir


# CUTLASS GPT
python curator.py --model=gpt2 --batch=1 --seq_len=512 --sm=$sm --precision=float32 --tmp_dir=$tmp_dir
python curator.py --model=gpt2 --batch=4 --seq_len=512 --sm=$sm --precision=float32 --tmp_dir=$tmp_dir
python curator.py --model=gpt2 --batch=8 --seq_len=512 --sm=$sm --precision=float32 --tmp_dir=$tmp_dir

# CUTLASS GPT-medium
python curator.py --model=gpt2-medium --batch=1 --seq_len=512 --sm=$sm --precision=float32 --tmp_dir=$tmp_dir
python curator.py --model=gpt2-medium --batch=4 --seq_len=512 --sm=$sm --precision=float32 --tmp_dir=$tmp_dir
python curator.py --model=gpt2-medium --batch=8 --seq_len=512 --sm=$sm --precision=float32 --tmp_dir=$tmp_dir


# CUTLASS openllama-3B
python curator.py --model=openlm-research/open_llama_3b --batch=1 --seq_len=512 --sm=$sm --precision=float32 --tmp_dir=$tmp_dir
python curator.py --model=openlm-research/open_llama_3b --batch=4 --seq_len=512 --sm=$sm --precision=float32 --tmp_dir=$tmp_dir
python curator.py --model=openlm-research/open_llama_3b --batch=8 --seq_len=512 --sm=$sm --precision=float32 --tmp_dir=$tmp_dir

# CUTLASS MetaLlama3-8B
python curator.py --model=meta-llama/Meta-Llama-3-8B-Instruct --batch=1 --seq_len=512 --sm=$sm --precision=float32 --tmp_dir=$tmp_dir
python curator.py --model=meta-llama/Meta-Llama-3-8B-Instruct --batch=4 --seq_len=512 --sm=$sm --precision=float32 --tmp_dir=$tmp_dir
python curator.py --model=meta-llama/Meta-Llama-3-8B-Instruct --batch=8 --seq_len=512 --sm=$sm --precision=float32 --tmp_dir=$tmp_dir


###############################################################################################

#CUTLASS BERT tiny
python curator.py --model=gaunernst/bert-tiny-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 FMHA --tmp_dir=$tmp_dir
python curator.py --model=gaunernst/bert-tiny-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 FMHA --tmp_dir=$tmp_dir
python curator.py --model=gaunernst/bert-tiny-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 FMHA --tmp_dir=$tmp_dir


# CUTLASS FMHA BERT mini
python curator.py --model=gaunernst/bert-mini-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 FMHA --tmp_dir=$tmp_dir
python curator.py --model=gaunernst/bert-mini-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 FMHA --tmp_dir=$tmp_dir
python curator.py --model=gaunernst/bert-mini-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 FMHA --tmp_dir=$tmp_dir

# CUTLASS FMHA BERT small
python curator.py --model=gaunernst/bert-small-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 FMHA --tmp_dir=$tmp_dir
python curator.py --model=gaunernst/bert-small-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 FMHA --tmp_dir=$tmp_dir
python curator.py --model=gaunernst/bert-small-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 FMHA --tmp_dir=$tmp_dir

# CUTLASS FMHA BERT medium
python curator.py --model=gaunernst/bert-medium-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 FMHA --tmp_dir=$tmp_dir
python curator.py --model=gaunernst/bert-medium-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 FMHA --tmp_dir=$tmp_dir
python curator.py --model=gaunernst/bert-medium-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 FMHA --tmp_dir=$tmp_dir

# CUTLASS FMHA BERT base
python curator.py --model=bert-base-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 FMHA --tmp_dir=$tmp_dir
python curator.py --model=bert-base-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 FMHA --tmp_dir=$tmp_dir
python curator.py --model=bert-base-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 FMHA --tmp_dir=$tmp_dir

# CUTLASS FMHA BERT large
python curator.py --model=bert-large-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 FMHA --tmp_dir=$tmp_dir
python curator.py --model=bert-large-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 FMHA --tmp_dir=$tmp_dir
python curator.py --model=bert-large-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 FMHA --tmp_dir=$tmp_dir


# CUTLASS FMHA GPT
python curator.py --model=gpt2 --batch=1 --seq_len=512 --sm=$sm --precision=float32 FMHA --tmp_dir=$tmp_dir
python curator.py --model=gpt2 --batch=4 --seq_len=512 --sm=$sm --precision=float32 FMHA --tmp_dir=$tmp_dir
python curator.py --model=gpt2 --batch=8 --seq_len=512 --sm=$sm --precision=float32 FMHA --tmp_dir=$tmp_dir

# CUTLASS FMHA GPT-medium
python curator.py --model=gpt2-medium --batch=1 --seq_len=512 --sm=$sm --precision=float32 FMHA --tmp_dir=$tmp_dir
python curator.py --model=gpt2-medium --batch=4 --seq_len=512 --sm=$sm --precision=float32 FMHA --tmp_dir=$tmp_dir
python curator.py --model=gpt2-medium --batch=8 --seq_len=512 --sm=$sm --precision=float32 FMHA --tmp_dir=$tmp_dir


# CUTLASS FMHA openllama-3B
python curator.py --model=openlm-research/open_llama_3b --batch=1 --seq_len=512 --sm=$sm --precision=float32 FMHA --tmp_dir=$tmp_dir
python curator.py --model=openlm-research/open_llama_3b --batch=4 --seq_len=512 --sm=$sm --precision=float32 FMHA --tmp_dir=$tmp_dir
python curator.py --model=openlm-research/open_llama_3b --batch=8 --seq_len=512 --sm=$sm --precision=float32 FMHA --tmp_dir=$tmp_dir

# CUTLASS FMHA MetaLlama3-8B
python curator.py --model=meta-llama/Meta-Llama-3-8B-Instruct --batch=1 --seq_len=512 --sm=$sm --precision=float32 FMHA --tmp_dir=$tmp_dir
python curator.py --model=meta-llama/Meta-Llama-3-8B-Instruct --batch=4 --seq_len=512 --sm=$sm --precision=float32 FMHA --tmp_dir=$tmp_dir
python curator.py --model=meta-llama/Meta-Llama-3-8B-Instruct --batch=8 --seq_len=512 --sm=$sm --precision=float32 FMHA --tmp_dir=$tmp_dir

###############################################################################################

#CUTLASS BERT tiny
python curator.py --model=gaunernst/bert-tiny-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python curator.py --model=gaunernst/bert-tiny-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python curator.py --model=gaunernst/bert-tiny-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir


# cuBLAs BERT mini
python curator.py --model=gaunernst/bert-mini-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python curator.py --model=gaunernst/bert-mini-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python curator.py --model=gaunernst/bert-mini-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir

# cuBLAs BERT small
python curator.py --model=gaunernst/bert-small-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python curator.py --model=gaunernst/bert-small-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python curator.py --model=gaunernst/bert-small-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir

# cuBLAs BERT medium
python curator.py --model=gaunernst/bert-medium-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python curator.py --model=gaunernst/bert-medium-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python curator.py --model=gaunernst/bert-medium-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir

# cuBLAs BERT base
python curator.py --model=bert-base-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python curator.py --model=bert-base-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python curator.py --model=bert-base-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir

# cuBLAs BERT large
python curator.py --model=bert-large-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python curator.py --model=bert-large-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python curator.py --model=bert-large-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir


# cuBLAs GPT
python curator.py --model=gpt2 --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python curator.py --model=gpt2 --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python curator.py --model=gpt2 --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir

# cuBLAs GPT-medium
python curator.py --model=gpt2-medium --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python curator.py --model=gpt2-medium --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python curator.py --model=gpt2-medium --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir


# cuBLAs openllama-3B
python curator.py --model=openlm-research/open_llama_3b --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python curator.py --model=openlm-research/open_llama_3b --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python curator.py --model=openlm-research/open_llama_3b --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir

# cuBLAs MetaLlama3-8B
python curator.py --model=meta-llama/Meta-Llama-3-8B-Instruct --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python curator.py --model=meta-llama/Meta-Llama-3-8B-Instruct --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python curator.py --model=meta-llama/Meta-Llama-3-8B-Instruct --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir

echo "Our Evaluation is Reported in curator/LLM/$tmp_dir text file"
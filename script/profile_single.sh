#!/bin/bash


sm=$1
tmp_dir=$2

cd ../LLM

echo "Our Evaluation is Reported in curator/LLM/$tmp_dir"

#CUTLASS BERT tiny
python profiler.py --model=gaunernst/bert-tiny-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir
python profiler.py --model=gaunernst/bert-tiny-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir
python profiler.py --model=gaunernst/bert-tiny-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir


# CUTLASS BERT mini
python profiler.py --model=gaunernst/bert-mini-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir
python profiler.py --model=gaunernst/bert-mini-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir
python profiler.py --model=gaunernst/bert-mini-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir

# CUTLASS BERT small
python profiler.py --model=gaunernst/bert-small-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir
python profiler.py --model=gaunernst/bert-small-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir
python profiler.py --model=gaunernst/bert-small-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir

# CUTLASS BERT medium
python profiler.py --model=gaunernst/bert-medium-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir
python profiler.py --model=gaunernst/bert-medium-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir
python profiler.py --model=gaunernst/bert-medium-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir

# CUTLASS BERT base
python profiler.py --model=bert-base-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir
python profiler.py --model=bert-base-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir
python profiler.py --model=bert-base-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir

# CUTLASS BERT large
python profiler.py --model=bert-large-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir
python profiler.py --model=bert-large-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir
python profiler.py --model=bert-large-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir


# CUTLASS GPT
python profiler.py --model=gpt2 --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir
python profiler.py --model=gpt2 --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir
python profiler.py --model=gpt2 --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir

# CUTLASS GPT-medium
python profiler.py --model=gpt2-medium --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir
python profiler.py --model=gpt2-medium --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir
python profiler.py --model=gpt2-medium --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir


# CUTLASS openllama-3B
python profiler.py --model=openlm-research/open_llama_3b --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir
python profiler.py --model=openlm-research/open_llama_3b --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir

if [ "$tmp_dir" = "./cutlass_a6000" ]; then

python profiler.py --model=openlm-research/open_llama_3b --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir

python profiler.py --model=meta-llama/Meta-Llama-3-8B-Instruct --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir
python profiler.py --model=meta-llama/Meta-Llama-3-8B-Instruct --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir
python profiler.py --model=meta-llama/Meta-Llama-3-8B-Instruct --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir

fi



###############################################################################################

# #CUTLASS BERT tiny
python profiler.py --model=gaunernst/bert-tiny-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS_FMHA --tmp_dir=$tmp_dir
python profiler.py --model=gaunernst/bert-tiny-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS_FMHA --tmp_dir=$tmp_dir
python profiler.py --model=gaunernst/bert-tiny-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS_FMHA --tmp_dir=$tmp_dir


# CUTLASS FMHA BERT mini
python profiler.py --model=gaunernst/bert-mini-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS_FMHA --tmp_dir=$tmp_dir
python profiler.py --model=gaunernst/bert-mini-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS_FMHA --tmp_dir=$tmp_dir
python profiler.py --model=gaunernst/bert-mini-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS_FMHA --tmp_dir=$tmp_dir

# CUTLASS FMHA BERT small
python profiler.py --model=gaunernst/bert-small-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS_FMHA --tmp_dir=$tmp_dir
python profiler.py --model=gaunernst/bert-small-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS_FMHA --tmp_dir=$tmp_dir
python profiler.py --model=gaunernst/bert-small-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS_FMHA --tmp_dir=$tmp_dir

# CUTLASS FMHA BERT medium
python profiler.py --model=gaunernst/bert-medium-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS_FMHA --tmp_dir=$tmp_dir
python profiler.py --model=gaunernst/bert-medium-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS_FMHA --tmp_dir=$tmp_dir
python profiler.py --model=gaunernst/bert-medium-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS_FMHA --tmp_dir=$tmp_dir

# CUTLASS FMHA BERT base
python profiler.py --model=bert-base-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS_FMHA --tmp_dir=$tmp_dir
python profiler.py --model=bert-base-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS_FMHA --tmp_dir=$tmp_dir
python profiler.py --model=bert-base-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS_FMHA --tmp_dir=$tmp_dir

# CUTLASS FMHA BERT large
python profiler.py --model=bert-large-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS_FMHA --tmp_dir=$tmp_dir
python profiler.py --model=bert-large-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS_FMHA --tmp_dir=$tmp_dir
python profiler.py --model=bert-large-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS_FMHA --tmp_dir=$tmp_dir


# CUTLASS FMHA GPT
python profiler.py --model=gpt2 --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS_FMHA --tmp_dir=$tmp_dir
python profiler.py --model=gpt2 --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS_FMHA --tmp_dir=$tmp_dir
python profiler.py --model=gpt2 --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS_FMHA --tmp_dir=$tmp_dir

# CUTLASS FMHA GPT-medium
python profiler.py --model=gpt2-medium --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS_FMHA --tmp_dir=$tmp_dir
python profiler.py --model=gpt2-medium --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS_FMHA --tmp_dir=$tmp_dir
python profiler.py --model=gpt2-medium --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS_FMHA --tmp_dir=$tmp_dir


# CUTLASS FMHA openllama-3B
python profiler.py --model=openlm-research/open_llama_3b --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS_FMHA --tmp_dir=$tmp_dir
python profiler.py --model=openlm-research/open_llama_3b --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS_FMHA --tmp_dir=$tmp_dir


if [ "$tmp_dir" = "./cutlass_a6000" ]; then

python profiler.py --model=openlm-research/open_llama_3b --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS_FMHA --tmp_dir=$tmp_dir

python profiler.py --model=meta-llama/Meta-Llama-3-8B-Instruct --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS_FMHA --tmp_dir=$tmp_dir
python profiler.py --model=meta-llama/Meta-Llama-3-8B-Instruct --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS_FMHA --tmp_dir=$tmp_dir
python profiler.py --model=meta-llama/Meta-Llama-3-8B-Instruct --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS_FMHA --tmp_dir=$tmp_dir


fi


###############################################################################################

#CUTLASS BERT tiny
python profiler.py --model=gaunernst/bert-tiny-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python profiler.py --model=gaunernst/bert-tiny-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python profiler.py --model=gaunernst/bert-tiny-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir


# cuBLAs BERT mini
python profiler.py --model=gaunernst/bert-mini-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python profiler.py --model=gaunernst/bert-mini-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python profiler.py --model=gaunernst/bert-mini-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir

# cuBLAs BERT small
python profiler.py --model=gaunernst/bert-small-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python profiler.py --model=gaunernst/bert-small-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python profiler.py --model=gaunernst/bert-small-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir

# cuBLAs BERT medium
python profiler.py --model=gaunernst/bert-medium-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python profiler.py --model=gaunernst/bert-medium-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python profiler.py --model=gaunernst/bert-medium-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir

# cuBLAs BERT base
python profiler.py --model=bert-base-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python profiler.py --model=bert-base-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python profiler.py --model=bert-base-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir

# cuBLAs BERT large
python profiler.py --model=bert-large-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python profiler.py --model=bert-large-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python profiler.py --model=bert-large-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir


# cuBLAs GPT
python profiler.py --model=gpt2 --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python profiler.py --model=gpt2 --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python profiler.py --model=gpt2 --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir

# cuBLAs GPT-medium
python profiler.py --model=gpt2-medium --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python profiler.py --model=gpt2-medium --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python profiler.py --model=gpt2-medium --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir


# cuBLAs openllama-3B
python profiler.py --model=openlm-research/open_llama_3b --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python profiler.py --model=openlm-research/open_llama_3b --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir

if [ "$tmp_dir" = "./cutlass_a6000" ]; then

python profiler.py --model=openlm-research/open_llama_3b --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir

# # cuBLAs MetaLlama3-8B
python profiler.py --model=meta-llama/Meta-Llama-3-8B-Instruct --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python profiler.py --model=meta-llama/Meta-Llama-3-8B-Instruct --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir
python profiler.py --model=meta-llama/Meta-Llama-3-8B-Instruct --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=cuBLAS --tmp_dir=$tmp_dir


fi

echo "Our Evaluation is Reported in curator/LLM/$tmp_dir"

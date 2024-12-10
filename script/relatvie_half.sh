#!/bin/bash


sm=$1
tmp_dir=$2

cd ../LLM

echo "Our Evaluation is Reported in curator/LLM/$tmp_dir text file"

#CUTLASS BERT tiny
python relatvie_performance.py --model=gaunernst/bert-tiny-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float16 --tmp_dir=$tmp_dir
python relatvie_performance.py --model=gaunernst/bert-tiny-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float16 --tmp_dir=$tmp_dir
python relatvie_performance.py --model=gaunernst/bert-tiny-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float16 --tmp_dir=$tmp_dir


# CUTLASS BERT mini
python relatvie_performance.py --model=gaunernst/bert-mini-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float16 --tmp_dir=$tmp_dir
python relatvie_performance.py --model=gaunernst/bert-mini-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float16 --tmp_dir=$tmp_dir
python relatvie_performance.py --model=gaunernst/bert-mini-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float16 --tmp_dir=$tmp_dir

# CUTLASS BERT small
python relatvie_performance.py --model=gaunernst/bert-small-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float16 --tmp_dir=$tmp_dir
python relatvie_performance.py --model=gaunernst/bert-small-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float16 --tmp_dir=$tmp_dir
python relatvie_performance.py --model=gaunernst/bert-small-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float16 --tmp_dir=$tmp_dir

# CUTLASS BERT medium
python relatvie_performance.py --model=gaunernst/bert-medium-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float16 --tmp_dir=$tmp_dir
python relatvie_performance.py --model=gaunernst/bert-medium-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float16 --tmp_dir=$tmp_dir
python relatvie_performance.py --model=gaunernst/bert-medium-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float16 --tmp_dir=$tmp_dir

# # CUTLASS BERT base
python relatvie_performance.py --model=bert-base-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float16 --tmp_dir=$tmp_dir
python relatvie_performance.py --model=bert-base-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float16 --tmp_dir=$tmp_dir
python relatvie_performance.py --model=bert-base-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float16 --tmp_dir=$tmp_dir

# CUTLASS BERT large
python relatvie_performance.py --model=bert-large-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float16 --tmp_dir=$tmp_dir
python relatvie_performance.py --model=bert-large-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float16 --tmp_dir=$tmp_dir
python relatvie_performance.py --model=bert-large-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float16 --tmp_dir=$tmp_dir


# CUTLASS GPT
python relatvie_performance.py --model=gpt2 --batch=1 --seq_len=512 --sm=$sm --precision=float16 --tmp_dir=$tmp_dir
python relatvie_performance.py --model=gpt2 --batch=4 --seq_len=512 --sm=$sm --precision=float16 --tmp_dir=$tmp_dir
python relatvie_performance.py --model=gpt2 --batch=8 --seq_len=512 --sm=$sm --precision=float16 --tmp_dir=$tmp_dir

# CUTLASS GPT-medium
python relatvie_performance.py --model=gpt2-medium --batch=1 --seq_len=512 --sm=$sm --precision=float16 --tmp_dir=$tmp_dir
python relatvie_performance.py --model=gpt2-medium --batch=4 --seq_len=512 --sm=$sm --precision=float16 --tmp_dir=$tmp_dir
python relatvie_performance.py --model=gpt2-medium --batch=8 --seq_len=512 --sm=$sm --precision=float16 --tmp_dir=$tmp_dir


# CUTLASS openllama-3B
python relatvie_performance.py --model=openlm-research/open_llama_3b --batch=1 --seq_len=512 --sm=$sm --precision=float16 --tmp_dir=$tmp_dir
python relatvie_performance.py --model=openlm-research/open_llama_3b --batch=4 --seq_len=512 --sm=$sm --precision=float16 --tmp_dir=$tmp_dir

if [ "$tmp_dir" = "./cutlass_a6000" ]; then
python relatvie_performance.py --model=openlm-research/open_llama_3b --batch=8 --seq_len=512 --sm=$sm --precision=float16 --tmp_dir=$tmp_dir

python relatvie_performance.py --model=meta-llama/Meta-Llama-3-8B-Instruct --batch=1 --seq_len=512 --sm=$sm --precision=float16 --tmp_dir=$tmp_dir
python relatvie_performance.py --model=meta-llama/Meta-Llama-3-8B-Instruct --batch=4 --seq_len=512 --sm=$sm --precision=float16 --tmp_dir=$tmp_dir
python relatvie_performance.py --model=meta-llama/Meta-Llama-3-8B-Instruct --batch=8 --seq_len=512 --sm=$sm --precision=float16 --tmp_dir=$tmp_dir
fi


###############################################################################################


echo "Our Relatvie is Reported in CURATOR_HOME/script csv file"
#!/bin/bash


sm=$1
tmp_dir=$2

cd ../LLM

echo "Our Evaluation is Reported in curator/LLM/$tmp_dir"

#CUTLASS BERT tiny
python figure_9.py --model=gaunernst/bert-tiny-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir
python figure_9.py --model=gaunernst/bert-tiny-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir
python figure_9.py --model=gaunernst/bert-tiny-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir


# CUTLASS BERT mini
python figure_9.py --model=gaunernst/bert-mini-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir
python figure_9.py --model=gaunernst/bert-mini-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir
python figure_9.py --model=gaunernst/bert-mini-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir

# CUTLASS BERT small
python figure_9.py --model=gaunernst/bert-small-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir
python figure_9.py --model=gaunernst/bert-small-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir
python figure_9.py --model=gaunernst/bert-small-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir

# # CUTLASS BERT medium
python figure_9.py --model=gaunernst/bert-medium-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir
python figure_9.py --model=gaunernst/bert-medium-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir
python figure_9.py --model=gaunernst/bert-medium-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir

# # CUTLASS BERT base
python figure_9.py --model=bert-base-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir
python figure_9.py --model=bert-base-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir
python figure_9.py --model=bert-base-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir

# # CUTLASS BERT large
python figure_9.py --model=bert-large-uncased --batch=1 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir
python figure_9.py --model=bert-large-uncased --batch=4 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir
python figure_9.py --model=bert-large-uncased --batch=8 --seq_len=512 --sm=$sm --precision=float32 --target_lib=CUTLASS --tmp_dir=$tmp_dir


cd ../LLM

# BERT tiny
python create_model.py --model=gaunernst/bert-tiny-uncased --batch=1 --seq_len=512
python create_model.py --model=gaunernst/bert-tiny-uncased --batch=4 --seq_len=512
python create_model.py --model=gaunernst/bert-tiny-uncased --batch=8 --seq_len=512

# BERT mini
python create_model.py --model=gaunernst/bert-mini-uncased --batch=1 --seq_len=512
python create_model.py --model=gaunernst/bert-mini-uncased --batch=4 --seq_len=512
python create_model.py --model=gaunernst/bert-mini-uncased --batch=8 --seq_len=512

# BERT small
python create_model.py --model=gaunernst/bert-small-uncased --batch=1 --seq_len=512
python create_model.py --model=gaunernst/bert-small-uncased --batch=4 --seq_len=512
python create_model.py --model=gaunernst/bert-small-uncased --batch=8 --seq_len=512

# BERT medium
python create_model.py --model=gaunernst/bert-medium-uncased --batch=1 --seq_len=512
python create_model.py --model=gaunernst/bert-medium-uncased --batch=4 --seq_len=512
python create_model.py --model=gaunernst/bert-medium-uncased --batch=8 --seq_len=512

# BERT base
python create_model.py --model=bert-base-uncased --batch=1 --seq_len=512
python create_model.py --model=bert-base-uncased --batch=4 --seq_len=512
python create_model.py --model=bert-base-uncased --batch=8 --seq_len=512

# BERT large
python create_model.py --model=bert-large-uncased --batch=1 --seq_len=512
python create_model.py --model=bert-large-uncased --batch=4 --seq_len=512
python create_model.py --model=bert-large-uncased --batch=8 --seq_len=512

##############################################################################################################################

# GPT2
python create_model.py --model=gpt2 --batch=1 --seq_len=512
python create_model.py --model=gpt2 --batch=4 --seq_len=512
python create_model.py --model=gpt2 --batch=8 --seq_len=512


# GPT2 medium
python create_model.py --model=gpt2-medium --batch=1 --seq_len=512
python create_model.py --model=gpt2-medium --batch=4 --seq_len=512
python create_model.py --model=gpt2-medium --batch=8 --seq_len=512

##############################################################################################################################

# openLlama-3B
python create_model.py --model=openlm-research/open_llama_3b --batch=1 --seq_len=512
python create_model.py --model=openlm-research/open_llama_3b --batch=4 --seq_len=512
python create_model.py --model=openlm-research/open_llama_3b --batch=8 --seq_len=512


# MetaLlama3-8B
# Put your HuggingFace read token

python create_model.py --model=meta-llama/Meta-Llama-3-8B-Instruct --batch=1 --seq_len=512 --token=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
python create_model.py --model=meta-llama/Meta-Llama-3-8B-Instruct --batch=4 --seq_len=512 --token=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
python create_model.py --model=meta-llama/Meta-Llama-3-8B-Instruct --batch=8 --seq_len=512 --token=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
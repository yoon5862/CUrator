import torch
from transformers import BertModel, GPT2Model, LlamaForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM
import transformers


import argparse
import onnx
import os

class TraceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, *inp):
        out = self.model(*inp)
        return out["logits"]

class CreateGpt2:
    def __init__(self, model, batch=1, seq_len=512):
        self.model_name = model
        self.batch_size = batch
        self.seq_len = seq_len
        
        assert self.seq_len <= 1024, "GPT2 Must be smaller then 1024"
        
        self.inputs = (torch.ones(self.batch_size, self.seq_len, dtype=torch.int64))
        
        self.dir_name = f"./onnx_file"
        if not os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)
            
    def forward(self):
        tmp = self.model_name.split("/")
        if len(tmp) > 1:
            file_tmp = tmp[1]
        else:
            file_tmp = tmp[0]
        
        model_full_name = f"{self.dir_name}/{file_tmp}_{self.batch_size}_{self.seq_len}.onnx"
        
        model = GPT2Model.from_pretrained(self.model_name)
        torch.onnx.export(model, self.inputs, model_full_name, True, input_names=["input_ids"])
        
        
class CreateBert:
    def __init__(self, model, batch=1, seq_len=512):
        self.model_name = model
        self.batch_size = batch
        self.seq_len = seq_len
        
        assert self.seq_len <= 512, "Bert Must be smaller then 512"
        
        #bert input
        self.inputs = (torch.ones(self.batch_size, self.seq_len, dtype=torch.int64),
                       torch.ones(self.batch_size, self.seq_len, dtype=torch.int64),
                       torch.ones(self.batch_size, self.seq_len, dtype=torch.int64))
        
        #create dir
        self.dir_name = "./onnx_file"
        if not os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)
        
    def forward(self):
        tmp = self.model_name.split("/")
        if len(tmp) > 1:
            file_tmp = tmp[1]
        else:
            file_tmp = tmp[0]
        
        model_full_name = f"{self.dir_name}/{file_tmp}_{self.batch_size}_{self.seq_len}.onnx"
        
        model = BertModel.from_pretrained(self.model_name)
        
        #bert input
        self.inputs = (torch.ones(self.batch_size, self.seq_len, dtype=torch.int64),
                       torch.ones(self.batch_size, self.seq_len, dtype=torch.int64),
                       torch.ones(self.batch_size, self.seq_len, dtype=torch.int64))
        
        torch.onnx.export(model, self.inputs, model_full_name, True, input_names=["input_ids", "attention_mask", "token_type_ids"])

class CreateLLama:
    def __init__(self, model, batch=1, seq_len=512):
        self.model_name = model
        self.batch_size = batch
        self.seq_len = seq_len
        
        assert self.seq_len <= 1024, "LLama3 Must be smaller then 1024"
        
        self.inputs = (torch.ones(self.batch_size, self.seq_len, dtype=torch.int64))
        
        tmp = self.model_name.split("/")
        if len(tmp) > 1:
            self.file_tmp = tmp[1]
        else:
            self.file_tmp = tmp[0]
        
        self.dir_name = f"./onnx_file/{self.file_tmp}_{self.batch_size}_{self.seq_len}"
        
        if not os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)
            
    def forward(self, token):
        model_full_name = f"{self.dir_name}/{self.file_tmp}_{self.batch_size}_{self.seq_len}.onnx"
        model = LlamaForCausalLM.from_pretrained(self.model_name, token=token, torch_dtype=torch.float32, return_dict=False) # put your own huggingface keys.
        torch.onnx.export(model, self.inputs, model_full_name, True, input_names=["input_ids"])
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="create onnx models")
    parser.add_argument('--batch', type=int, default=1, help="Batch size")
    parser.add_argument('--seq_len', type=int, default=512, help="Sequence length")
    parser.add_argument('--model', type=str, default="gaunernst/bert-tiny-uncased", help="Model to export onnx")
    parser.add_argument('--token', type=str, required=False, help="put huggingface read token")
    
    args = parser.parse_args()
    batch=args.batch
    seq_len=args.seq_len
    model = args.model
    
    print("ONNX model creating")
    print(f"Model: {model}")
    print(f"input(Batch, Sequence length): ({batch}, {seq_len})")
    
    if "bert" in model:
        bert = CreateBert(model=model, batch=batch, seq_len=seq_len)
        bert.forward()
    elif "gpt2" in model:
        gpt2 = CreateGpt2(model=model, batch=batch, seq_len=seq_len)
        gpt2.forward()
    elif "llama" in model or "Llama" in model:
        llama3 = CreateLLama(model=model, batch=batch, seq_len=seq_len)
        llama3.forward(args.token)
    
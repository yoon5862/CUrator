import os

# tvm library
import tvm
from tvm import relay
from tvm.relay.transform import ToMixedPrecision


# onnx library
import onnx

class LoadLLama3:
    def __init__(self, model, batch=1, seq_len=512, mixedPrecision=False, load_external_data=True):
        self.model = model
        self.batch = batch
        self.seq_len = seq_len
        self.mixedPrecision = mixedPrecision
        self.load_external_data = load_external_data
        
        assert self.seq_len <= 1024, "Sequence length must small and equal than 1024"
        
        split_model_name = self.model.split("/")
        if len(split_model_name) > 1:
            model_file = split_model_name[1]
        else:
            model_file = split_model_name[0]
        
        # onnx_file = f"/scratch2/5862www/onnx_file/{model_file}_{self.batch}_{self.seq_len}/{model_file}_{self.batch}_{self.seq_len}.onnx"
        onnx_file = f"./onnx_file/{model_file}_{self.batch}_{self.seq_len}/{model_file}_{self.batch}_{self.seq_len}.onnx"
        assert os.path.exists(onnx_file), f"ONNX File is not exist\nCreate onnx file running create_model.py {onnx_file}"
        
        # onnx -> tvm.relay
        onnx_model = onnx.load(onnx_file, load_external_data=self.load_external_data)
        self.mod, self.params = relay.frontend.from_onnx(onnx_model, {"input_ids":(self.batch, self.seq_len)})
        
        if self.mixedPrecision == True:
            self.mod = ToMixedPrecision("float16")(self.mod)
    
    def getMod(self):
        return self.mod
    
    def getParams(self):
        return self.params
    
    def getModels(self):
        return self.mod, self.params
    
    def get_input_dimension(self):
        return (self.batch, self.seq_len)

class LoadGPT2:
    def __init__(self, model, batch=1, seq_len=512, mixedPrecision=False, load_external_data=False):
        self.model = model
        self.batch = batch
        self.seq_len = seq_len
        self.mixedPrecision = mixedPrecision
        self.load_external_data = load_external_data
        
        assert self.seq_len <= 1024, "Sequence length must small and equal than 1024"
        
        split_model_name = self.model.split("/")
        if len(split_model_name) > 1:
            model_file = split_model_name[1]
        else:
            model_file = split_model_name[0]
        
        onnx_file = f"./onnx_file/{model_file}_{self.batch}_{self.seq_len}.onnx"
        assert os.path.exists(onnx_file), "ONNX File is not exist\nCreate onnx file running create_model.py"
        
        # onnx -> tvm.relay
        onnx_model = onnx.load(onnx_file, load_external_data=self.load_external_data)
        self.mod, self.params = relay.frontend.from_onnx(onnx_model, {"input_ids":(self.batch, self.seq_len)})
        
        if self.mixedPrecision == True:
            self.mod = ToMixedPrecision("float16")(self.mod)
    
    def getMod(self):
        return self.mod
    
    def getParams(self):
        return self.params
    
    def getModels(self):
        return self.mod, self.params
    
    def get_input_dimension(self):
        return (self.batch, self.seq_len)
      
class LoadBERT:
    def __init__(self, model, batch=1, seq_len=512, mixedPrecision=False, load_external_data=False):
        self.model = model
        self.batch = batch
        self.seq_len = seq_len
        self.mixedPrecision = mixedPrecision
        self.load_external_data = load_external_data
        
        assert self.seq_len <= 512, "Sequence length should small and equal than 512"
        
        split_model_name = self.model.split("/")
        if len(split_model_name) > 1:
            model_file = split_model_name[1]
        else:
            model_file = split_model_name[0]
        
        onnx_file = f"onnx_file/{model_file}_{self.batch}_{self.seq_len}.onnx"
        assert os.path.exists(onnx_file), "ONNX File is not exist\nCreate onnx file running create_model.py"
        
        # onnx -> tvm.relay
        onnx_model = onnx.load(onnx_file, load_external_data=self.load_external_data)
        self.mod, self.params = relay.frontend.from_onnx(onnx_model, {"input_ids":(self.batch, self.seq_len), "attention_mask": (self.batch, self.seq_len), "token_type_ids": (self.batch, self.seq_len)})
        
        if self.mixedPrecision == True:
            self.mod = ToMixedPrecision("float16")(self.mod)
        
    def getMod(self):
        return self.mod
    
    def getParams(self):
        return self.params
    
    def getModels(self):
        return self.mod, self.params
    
    def get_input_dimension(self):
        return ((self.batch, self.seq_len), (self.batch, self.seq_len), (self.batch, self.seq_len))
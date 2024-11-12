import tvm
from tvm.contrib import graph_executor
from loadLLM import LoadBERT, LoadGPT2, LoadLLama3
from tvm.contrib import curator
from tvm.contrib import cutlass
import argparse
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End inference of LLM's First Token")
    parser.add_argument('--model', type=str, default="gaunernst/bert-tiny-uncased", help="Model from onnx file")
    parser.add_argument('--batch', type=int, default=1, help="Batch size: 1/4/8")
    parser.add_argument('--seq_len', type=int, default=512, help="Sequence length: 512")
    parser.add_argument('--sm', type=int, default=80, help="GPU Architecture: 70/80/86/89")
    parser.add_argument('--precision', type=str, default="float32", help="Support precision: float32/float16")
    parser.add_argument('--tmp_dir', type=str, default="./rtx4090", help="The tmp_dir must be set differently for each GPU")
    parser.add_argument('--target_lib', type=str, default="CUrator", help="Support library: Ansor/BOLT/cuBLAS/CUTLASS/CUrator")
    
    args = parser.parse_args()
    model = args.model
    batch = args.batch
    seq_len = args.seq_len
    mixedPrecision = False if args.precision == "float32" else True
    sm=int(args.sm)
    tmp_dir = args.tmp_dir
    target_lib = args.target_lib
    
    inference_rlt = f"{model}_{batch}_{seq_len}_{args.precision}.json"
    inference_rlt = inference_rlt.replace("/", "-")
    inference_rlt_dir = os.path.join(tmp_dir, inference_rlt)
    
    if os.path.exists(inference_rlt_dir):
        json_file = {}
        
        with open(inference_rlt_dir, "r") as file:
            for line in file:
                json_data = json.loads(line.strip())
                assert json_data["target_lib"] != target_lib, "Library already Profiled!"
    
    # set tuning parameter 
    host = tvm.target.Target("llvm")
    curator_target = {
            "kind":"curator",
            "sm": sm,
            "tbt_m": [32, 256, 32],
            "tbt_n": [32, 256, 32],
            "tbt_k": [32, 32, 32] if mixedPrecision == False else [32, 64, 32],
            "pipelining_range": [2, 2, 2] if mixedPrecision == False else [4, 10, 2],
            "split_k_range": [2, 8, 2],
            "swizzle_range": [8],
            "use_fast_math": True,
            "tmp_dir": tmp_dir,
        }
    
    if mixedPrecision == True:
        curator_target["align_range"] = [1, 2, 4, 8]
    
    if "bert" in model:
        bert = LoadBERT(model, batch=batch, seq_len=seq_len, mixedPrecision=mixedPrecision)
        mod, params = bert.getModels()
    elif "llama" in model:
        llama3 = LoadLLama3(model, batch=batch, seq_len=seq_len, mixedPrecision=mixedPrecision)
        mod, params = llama3.getModels()
    elif "gpt2" in model:
        gpt2 = LoadGPT2(model, batch=batch, seq_len=seq_len, mixedPrecision=mixedPrecision)
        mod, params = gpt2.getModels()
    
    host = tvm.target.Target("llvm")
    cuda = tvm.target.Target("cuda", host=host)
    dev = tvm.device(str(cuda), 0)
    
    print(f"Model: {model}")
    print(f"Input: ({batch}, {seq_len})")
    
    inference_time = 0.0
    if "cuBLAS" in target_lib:
        cublas_module = curator.cublas_module(mod, params)
        cublas_rlt = cublas_module.benchmark(dev, number=2, repeat=10)
        print(f"cuBLAS: {cublas_rlt.mean * 1000} ms")
        inference_time = cublas_rlt.mean * 1000
    elif "Ansor" in target_lib:
        ansor_module = curator.ansor_module(mod, params, curator_target=curator_target, model=model, batch=batch, seq_len=seq_len, mixedPrecision=mixedPrecision)
        ansor_rlt = ansor_module.benchmark(dev, number=2, repeat=10)
        print(f"Ansor: {ansor_rlt.mean * 1000} ms")
        inference_time = ansor_rlt.mean * 1000
    elif "BOLT" in target_lib:
        bolt_module = curator.bolt_module(mod, params, sm, "./bolt")
        bolt_rlt = bolt_module.benchmark(dev, number=2, repeat=10)
        print(f"BOLT: {bolt_rlt.mean * 1000} ms")
        inference_time = bolt_rlt.mean * 1000
    elif "CUTLASS_FMHA" in target_lib:
        cutlass_module = curator.cutlass_module_fmha(mod, params, curator_target)
        cutlass_rlt = cutlass_module.benchmark(dev, number=2, repeat=10)
        print(f"CUTLASS w/ FMHA: {cutlass_rlt.mean * 1000} ms")
        inference_time = cutlass_rlt.mean
    elif "CUTLASS" in target_lib:
        cutlass_module = curator.cutlass_module_natural(mod, params, curator_target)
        cutlass_rlt = cutlass_module.benchmark(dev, number=2, repeat=10)
        print(f"CUTLASS w/o FMHA: {cutlass_rlt.mean * 1000} ms")
        inference_time = cutlass_rlt.mean * 1000
    
    print(f"Recording in {inference_rlt_dir}")
    
    json_info = {"target_lib": target_lib, "inference": inference_time}
    with open(inference_rlt_dir, "a") as file:
        json.dump(json_info, file)
        file.write("\n")
    
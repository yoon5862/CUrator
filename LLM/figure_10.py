import tvm
from tvm.contrib import graph_executor
from loadLLM import LoadBERT, LoadGPT2, LoadLLama3
from tvm.contrib import curator
from tvm.contrib import cutlass
import argparse
import json
import os
import pandas as pd
from tvm import relay
import numpy as np


def create_batchMatmul(batch, M, N, K, dtype="float32"):
    a = relay.var("lhs_input", shape=(batch, M, K), dtype=dtype)
    b = relay.var("rhs_input", shape=(batch, N, K), dtype=dtype)
    
    matmul = relay.nn.batch_matmul(a, b, out_dtype=dtype, transpose_a=False, transpose_b=True)
    
    mod = tvm.IRModule.from_expr(matmul)
    mod = relay.transform.InferType()(mod)
    
    b_arr = np.array([1 for _ in range(batch * N * K)]).reshape(batch, N, K).astype(dtype)
    params = {"rhs_input": b_arr}
    
    return mod, params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End inference of LLM's First Token")
    parser.add_argument('--batch', type=int, default=1, help="Batch size: 1/4/8")
    parser.add_argument('--sm', type=int, default=80, help="GPU Architecture: 70/80/86/89")
    parser.add_argument('--precision', type=str, default="float32", help="Support precision: float32/float16")
    parser.add_argument('--tmp_dir', type=str, default="./rtx4090", help="The tmp_dir must be set differently for each GPU")
    parser.add_argument('--target_lib', type=str, default="CUrator", help="Support library: Ansor/BOLT/cuBLAS/CUTLASS/CUrator")
    
    args = parser.parse_args()
    batch = args.batch
    mixedPrecision = False if args.precision == "float32" else True
    sm=int(args.sm)
    tmp_dir = args.tmp_dir
    target_lib = args.target_lib
    
    print("figure10")
    print(f"{batch}, 512, 64, 512")
    
    inference_rlt = f"{batch}_{args.precision}.json"
    inference_rlt = inference_rlt.replace("/", "-")
    inference_rlt_dir = os.path.join(tmp_dir, inference_rlt)
    
    lib_inference = {}
    assert os.path.exists(inference_rlt_dir), "[Error] Please run figure10.sh with \"git checkout moduleTest_21\""
    if os.path.exists(inference_rlt_dir):
        json_file = []
        
        with open(inference_rlt_dir, "r") as file:
            for line in file:
                json_data = json.loads(line.strip())
                # assert json_data["target_lib"] != target_lib, "Library already Profiled!"
                json_file.append(json_data["target_lib"])
                lib_inference[json_data["target_lib"]] = json_data["inference"]
        
        assert "CUTLASS" in json_file and "cuBLAS" in json_file, "[Error] Please run figure10.sh with \"git checkout moduleTest_21\""
    
    if "CUrator" in lib_inference.keys():
      del lib_inference["CUrator"]
    if "CUTLASS_FMHA" in lib_inference.keys():
      del lib_inference["CUTLASS_FMHA"]
    
    
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
    
    
    mod, params = create_batchMatmul(batch, 512, 64, 512)
    
    host = tvm.target.Target("llvm")
    cuda = tvm.target.Target("cuda", host=host)
    dev = tvm.device(str(cuda), 0)
    
    
    model = "gpt2"
    seq_len = 512
    inference_time = 0.0
    if "cuBLAS" in target_lib:
        cublas_module = curator.cublas_module(mod, params)
        cublas_rlt = cublas_module.benchmark(dev, number=20, repeat=100)
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
    elif "CUTLASS_FMHA" in target_lib and sm >= 80:
        cutlass_module = curator.cutlass_module_fmha(mod, params, curator_target, model)
        cutlass_rlt = cutlass_module.benchmark(dev, number=2, repeat=10)
        print(f"CUTLASS w/ FMHA: {cutlass_rlt.mean * 1000} ms")
        inference_time = cutlass_rlt.mean * 1000
    elif "CUTLASS" in target_lib:
        cutlass_module = curator.cutlass_module_natural(mod, params, curator_target, model)
        cutlass_rlt = cutlass_module.benchmark(dev, number=20, repeat=100)
        print(f"CUTLASS w/o FMHA & Figure 10: {cutlass_rlt.mean * 1000} ms")
        inference_time = cutlass_rlt.mean * 1000
    
    
    figure_10_csv = "figure_10.csv"
    figure_10_dir = os.path.join("./", figure_10_csv)
    
    
    cublas_time = lib_inference["cuBLAS"]
    cutlass_time = lib_inference["CUTLASS"]
    
    csv_data = {}
    csv_data["dimension"] = f"{batch}_512_64_512"
    csv_data["figure10_(a)"] = inference_time / cutlass_time
    csv_data["figure10_(b)"] = cublas_time / cutlass_time
    
    df_log = pd.DataFrame([csv_data])
    df_log = df_log[["dimension", "figure10_(a)", "figure10_(b)"]]

    if os.path.exists(figure_10_dir):
        df_log.to_csv(figure_10_dir, mode='a', header=False, index=False)
    else:
        df_log.to_csv(figure_10_dir, index=False)
    
    
    
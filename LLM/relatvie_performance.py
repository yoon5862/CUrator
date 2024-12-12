from tvm.contrib import cutlass
import argparse
import json
import os
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End inference of LLM's First Token")
    parser.add_argument('--model', type=str, default="gaunernst/bert-tiny-uncased", help="Model from onnx file")
    parser.add_argument('--batch', type=int, default=1, help="Batch size: 1/4/8")
    parser.add_argument('--seq_len', type=int, default=512, help="Sequence length: 512")
    parser.add_argument('--sm', type=int, default=80, help="GPU Architecture: 70/80/86/89")
    parser.add_argument('--precision', type=str, default="float32", help="Support precision: float32/float16")
    parser.add_argument('--tmp_dir', type=str, default="./rtx4090", help="The tmp_dir must be set differently for each GPU")
    
    args = parser.parse_args()
    model = args.model
    batch = args.batch
    seq_len = args.seq_len
    mixedPrecision = False if args.precision == "float32" else True
    sm=int(args.sm)
    tmp_dir = args.tmp_dir
    
    
    inference_rlt = f"{model}_{batch}_{seq_len}_{args.precision}.json"
    inference_rlt = inference_rlt.replace("/", "-")
    inference_rlt_dir = os.path.join(tmp_dir, inference_rlt)
    
    end_to_end_rlt = f"{model}_{args.precision}.txt"
    end_to_end_rlt = end_to_end_rlt.replace("/", "-")
    end_to_end_rlt_dir = os.path.join(tmp_dir, end_to_end_rlt)
    
    
    profiled_lib = []
    with open(inference_rlt_dir, "r") as file:
      for line in file:
        json_data = json.loads(line.strip())
        profiled_lib.append(json_data)
    
    profiled_datas = {}
    for profiled_data in profiled_lib:
      target_lib = profiled_data["target_lib"]
      inference_time = profiled_data["inference"]    
      profiled_datas[target_lib] = inference_time
    
    cublas_performance = profiled_datas["cuBLAS"]
    for key, value in profiled_datas.items():
      profiled_datas[key] = cublas_performance / profiled_datas[key]
    profiled_datas["name"] = f"{model}_{batch}_{seq_len}"
    
    
    precision_ = ""
    # open csv
    if args.precision == "float32":
      precision_ = "figure_7"
    elif args.precision == "float16":
      precision_ = "figure_8"
    
    inference_rlt = f"{precision_}.csv"
    inference_dir = os.path.join("./", inference_rlt)
    
    df_log = pd.DataFrame([profiled_datas])
    df_log = df_log[['name', 'cuBLAS', "CUTLASS", "CUTLASS_FMHA", "CUrator"]]
    print(df_log)
    
    if os.path.exists(inference_dir):
      df_log.to_csv(inference_dir, mode='a', header=False, index=False)
    else:
      df_log.to_csv(inference_dir, index=False)

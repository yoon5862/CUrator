from tvm.contrib import cutlass
import matplotlib.pyplot as plt
import argparse
import json
import os
import numpy as np
import pandas as pd


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="End-to-End inference of LLM's First Token")
  parser.add_argument('--precision', type=str, default="float32", help="Support precision: float32/float16")
  parser.add_argument('--tmp_dir', type=str, default="./rtx4090", help="The tmp_dir must be set differently for each GPU")
  
  args = parser.parse_args()
  precision = args.precision
  tmp_dir = args.tmp_dir
  
  precision_ = ""
  # open csv
  if args.precision == "float32":
    precision_ = "figure_7"
  elif args.precision == "float16":
    precision_ = "figure_8"
  
  inference_rlt = f"{precision_}.csv"
  inference_dir = os.path.join(tmp_dir, inference_rlt)
  
  assert os.path.exists(inference_dir)
  
  
  df = pd.read_csv(inference_dir)
  
  w = 0.20
  idx = np.arange(len(df)) 
  plt.figure(figsize=(30, 15)) 
  
  
  plt.bar(idx - 2 * w, df["cuBLAS"], width=w, edgecolor='black', label="cuBLAS")
  plt.bar(idx - w, df["CUTLASS"], width=w, edgecolor='black', label="CUTLASS w/o FMHA")
  plt.bar(idx, df["CUTLASS_FMHA"], width=w, edgecolor='black', label="CUTLASS w/ FMHA")
  plt.bar(idx + w, df["CUrator"], width=w, edgecolor='black', label="CUrator")
  plt.xticks(idx - w * 0.5, df["name"], rotation=-90)
  plt.xticks(fontsize=30)
  plt.yticks(fontsize=30)
  
  plt.legend(fontsize=20, ncol=4)
  plt.tight_layout()
  
  plt.savefig(f"{tmp_dir}/{precision_}.png", dpi=300)
  
  
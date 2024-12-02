from .library import *
# from library import *
import multiprocessing
import json
import os


def instantiate_fmha_template(attrs, func_args, torch_var1, func_name):
  template = """
    //cutlass support
    using ElementInputA = ${data_type};
    const int batch = ${batch};
    const int seq_len = ${seq_len};
    const int head_num = ${head_num};
    const int head_size = ${head_size};
    
    ${query}
    ${key}
    ${value}
    void *ptr_out = (void*)(out0->data);
    
    using Attention = AttentionKernel<
      ElementInputA,
      cutlass::arch::Sm80,
      ${alignment},
      ${kQueriesPerBlock},
      ${kKeysPerBlock},
      ${kMaxK},
      false,
      false
    >;
    
    typename Attention::Params p;
    {
      p.query_ptr = (ElementInputA*)q;
      p.key_ptr = (ElementInputA*)k;
      p.value_ptr = (ElementInputA*)v;
      p.logsumexp_ptr = nullptr;
      ${output_accum_ptr}
      
      p.output_ptr = (ElementInputA*)ptr_out;
      p.scale = 1.0f / sqrt(float(head_size));
      
      p.num_heads = head_num;
      p.num_batches = batch;
      p.head_dim = head_size;
      p.head_dim_value = head_size;
      p.num_queries = seq_len;
      p.num_keys = seq_len;
      p.custom_mask_type = Attention::CausalFromTopLeft;

      p.q_strideH = head_size;
      p.k_strideH = head_size;
      p.v_strideH = head_size;

      p.q_strideM = int32_t(head_num * head_size);
      p.k_strideM = int32_t(head_num * head_size);
      p.v_strideM = int32_t(head_num * head_size);

      p.q_strideB = p.q_strideM * seq_len;
      p.k_strideB = p.k_strideM * seq_len;
      p.v_strideB = p.v_strideM * seq_len;
    }
    
    constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
    int smem_bytes = sizeof(typename Attention::SharedStorage);

    cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);
    """
  
  aux_map = {}
  new_attrs = {}
  
  dimension = [str(int(dim)) for dim in attrs["qkv_shape"]]
  dtype = str(attrs["ElementInputA"])
  aux_map.update({"output_accum_ptr": "p.logsumexp_ptr = nullptr;"}) if len(func_args) < 7 else aux_map.update({"output_accum_ptr": "p.output_accum_ptr = (void*)(${arg6}->data);\n"})
  
  aux_map["batch"] = dimension[0]
  aux_map["seq_len"] = dimension[1]
  aux_map["head_num"] = dimension[2]
  aux_map["head_size"] = dimension[3]
  aux_map["alignment"] = "true" if (int(dimension[3]) % 8) == 0 and dtype == "cutlass::half_t" else "false" 
  aux_map["data_type"] = dtype
  
  
  aux_map["kQueriesPerBlock"] = str(attrs["kQueriesPerBlock"])
  aux_map["kKeysPerBlock"] = str(attrs["kKeysPerBlock"])
  aux_map["kMaxK"] = str(attrs["kMaxK"])
  
  # void *q = (void*)(${arg0}->data);
  # void *k = (void*)(${arg2}->data);
  # void *v = (void*)(${arg5}->data);
  
  aux_map["query"] = "void *q = (void*)(${arg0}->data);"
  aux_map["key"] = "void *k = (void*)(${arg2}->data);"
  aux_map["value"] = "void *v = (void*)(${arg5}->data);"
  
  if "curator.fmha2_bert_fp32" in func_name:
    aux_map["query"] = "void *q = (void*)(${arg0}->data);"
    aux_map["key"] = "void *k = (void*)(${arg1}->data);"
    aux_map["value"] = "void *v = (void*)(${arg2}->data);"
  if "curator.fmha2_bert_fp16" in func_name:
    aux_map["query"] = "void *q = (void*)(${arg0}->data);"
    aux_map["key"] = "void *k = (void*)(${arg1}->data);"
    aux_map["value"] = "void *v = (void*)(${arg4}->data);"
  if "curator.fmha2_gpt_fp32" in func_name:
    aux_map["query"] = "void *q = (void*)(${arg1}->data);"
    aux_map["key"] = "void *k = (void*)(${arg2}->data);"
    aux_map["value"] = "void *v = (void*)(${arg3}->data);"
  if "curator.fmha2_gpt_fp16" in func_name:
    aux_map["query"] = "void *q = (void*)(${arg1}->data);"
    aux_map["key"] = "void *k = (void*)(${arg2}->data);"
    aux_map["value"] = "void *v = (void*)(${arg4}->data);"
  
  template = substitute_template(template, aux_map)
  
  for i, arg in enumerate(func_args):
    new_attrs["arg{}".format(i)] = arg
  
  # template = substitute_template(template, new_attrs)
  return substitute_template(template, new_attrs)


def instantiate_fmha_template_flash(attrs, func_args):
  template = """
    //cutlass support
    using ElementInputA = cutlass::half_t;
    const int batch = ${batch};
    const int seq_len = ${seq_len};
    const int head_num = ${head_num};
    const int head_size = ${head_size};
    
    void *q = (void*)(${arg0}->data);
    void *k = (void*)(${arg2}->data);
    void *v = (void*)(${arg5}->data);
    void *softmax_lse = (void*)(${arg6}->data);
    // void *softmax_lse_accum = (void*)(${arg7}->data);
    // void *out_accum = (void*)(${arg8}->data);
    
    
    void *ptr_out = (void*)(out0->data);
    
    const int window_size_left = -1, window_size_right = 0;
    const float softcap = 0.0f;
    const float softmax_scale = sqrt(float(head_size));
    
    Flash_fwd_params params;
    set_params_fprop(params,
                     batch,
                     seq_len, seq_len,
                     seq_len, seq_len,
                     head_num, head_num,
                     head_num, head_num,
                     q, k, v, ptr_out,
                     softmax_lse,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap, head_size);
    
    params.num_splits = 1;
    //  params.softmax_lseaccum_ptr = softmax_lse_accum;
    // params.oaccum_ptr = out_accum;
    // run_mha_fwd64_64(params);
    
    run_mha_fwd_hdim128_32<cutlass::half_t, 128, true>(params);
    
    // run_mha_fwd_splitk(params);
    """
  
  aux_map = {}
  new_attrs = {}
  
  dimension = [str(int(dim)) for dim in attrs["qkv_shape"]]
  
  aux_map["batch"] = dimension[0]
  aux_map["seq_len"] = dimension[1]
  aux_map["head_num"] = dimension[2]
  aux_map["head_size"] = dimension[3]
  
  
  template = substitute_template(template, aux_map)
  
  for i, arg in enumerate(func_args):
    new_attrs["arg{}".format(i)] = arg
  
  # template = substitute_template(template, new_attrs)
  return substitute_template(template, new_attrs)


def tuning_fmha_func(head_size, dtype):
  template = """
  #include<iostream>
  #include <unistd.h>
  #include<string>
  #include<fstream>
  #include "cutlass/cutlass.h"
  #include "kernel_forward.h"
  using namespace std;

  #ifndef Q_TILE
  #define Q_TILE 64
  #endif

  #ifndef K_TILE
  #define K_TILE 64
  #endif

  #ifndef KMAX_TILE
  #define KMAX_TILE 64
  #endif

  int main(int argc, char *argv[]){
    int batch = 1;
    int seq_len = 512;
    int head_num = 12;
    int head_size = 64;
    std::string rlt_json_dir = "";

    int option;
    while((option = getopt(argc, argv, "b:s:n:h:d:")) != -1){
        switch(option){
            case 'b':
                batch = std::stoi(optarg);
                break;
            case 's':
                seq_len = std::stoi(optarg);
                break;
            case 'n':
                head_num = std::stoi(optarg);
                break;
              case 'h':
                head_size = std::stoi(optarg);
                break;
              case 'd':
              rlt_json_dir = std::string(optarg);
              break;
            case '?':
                break;
        }
    }

    // std::cout << batch << " " << seq_len << " " << head_num << " " << head_size << std::endl;

    ${data_type} *q, *k, *v, *o;

    const size_t data_size = batch * seq_len * head_num * head_size;

    cudaMalloc(&q, sizeof(${data_type}) * data_size);
    cudaMalloc(&k, sizeof(${data_type}) * data_size);
    cudaMalloc(&v, sizeof(${data_type}) * data_size);
    cudaMalloc(&o, sizeof(${data_type}) * data_size);

    cudaMemset(q, 1, sizeof(${data_type}) * data_size);
    cudaMemset(k, 1, sizeof(${data_type}) * data_size);
    cudaMemset(v, 1, sizeof(${data_type}) * data_size);
    cudaMemset(o, 1, sizeof(${data_type}) * data_size);

    
    static constexpr int kMaxK = KMAX_TILE;
    static int const kQueriesPerBlock = Q_TILE;
    static int const kKeysPerBlock = K_TILE;

    using Attention = AttentionKernel<
    ${data_type},
    cutlass::arch::Sm80,
    ${alignment},
    kQueriesPerBlock,
    kKeysPerBlock,
    kMaxK,
    false,
    false
    >;


    typename Attention::Params p;
    {
      p.query_ptr = q;
      p.key_ptr = k;
      p.value_ptr = v;
      p.logsumexp_ptr = nullptr;
      p.output_accum_ptr = nullptr;

      if(Attention::kNeedsOutputAccumulatorBuffer){
        cudaMalloc(&p.output_accum_ptr, sizeof(${data_type}) * data_size);
      }
      p.output_ptr = o;
      p.scale = 1.0f / sqrt(${data_type}(head_size));

      p.num_heads = head_num;
      p.num_batches = batch;
      p.head_dim = head_size;
      p.head_dim_value = head_size;
      p.num_queries = seq_len;
      p.num_keys = seq_len;
      p.custom_mask_type = Attention::CausalFromTopLeft;

      p.q_strideH = head_size;
      p.k_strideH = head_size;
      p.v_strideH = head_size;

      p.q_strideM = int32_t(head_num * head_size);
      p.k_strideM = int32_t(head_num * head_size);
      p.v_strideM = int32_t(head_num * head_size);

      p.q_strideB = p.q_strideM * seq_len;
      p.k_strideB = p.k_strideM * seq_len;
      p.v_strideB = p.v_strideM * seq_len;
    }


    constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
    int smem_bytes = sizeof(typename Attention::SharedStorage);

    cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    if(!Attention::check_supported(p)){
      std::cerr << "Kernel does not support these inputs" << std::endl;
      return 0;
    }

    for(int i = 0; i < 10; i++) kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);

    cudaEvent_t start, end;
    float elasped_time = 0.0f;

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    for(int i = 0; i < 10; i++) kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&elasped_time, start, end);
    
    std::fstream dataFile;
    std::string fileName = rlt_json_dir + "/"  + std::to_string(batch) + "_" + std::to_string(seq_len) + "_" + std::to_string(head_num) + "_" + std::to_string(head_size) + ".json";
    std::string json = "{\\"dim\\": [" + std::to_string(kQueriesPerBlock) + ", " + std::to_string(kKeysPerBlock) + ", " + std::to_string(kMaxK) +  "], " + "\\"time\\": " + std::to_string(elasped_time / 10) +  "}";

    dataFile.open(fileName, std::ios::app);
    dataFile << json << std::endl;

    // std::cout << json << std::endl;

    return 0;
  }
  """
  
  aux_map = {}
  aux_map["data_type"] = "cutlass::half_t" if dtype == "float16" else "float" 
  aux_map["alignment"] = "true" if ((head_size) % 8) == 0 and dtype == "float16" else "false" 
  template = substitute_template(template, aux_map)
  
  return template
  
  


class ProfileFMHA:
  def __init__(self, sm=89, path="./cutlass"):
    self.cache = {}
    self.sm = sm
    self.path = path
    
    self.batch = 1
    self.seq_len = 512
    self.head_num = 12
    self.head_size = 64
    self.dtype = "float16"
    
    self.real_path = os.path.dirname(__file__)
    # self.real_path = "/home/5862www/curator/tvm/python/tvm/contrib/curator"
    self.cutlass_path = f"{self.real_path}/../../../../3rdparty/cutlass/include/support"
    
    self.code_path = self.cutlass_path + f"/flash_attn/cutlass_flash.cu"
    self.fmha_cutlass_path = self.cutlass_path + "/cutlass-3.5.1"
    
    self.object_dir = self.path + "/src_cutlass_fmha"
    self.rlt_dir = self.path + "/rlt_cutlass_fmha"
    
    # if not os.path.exists(self.object_dir):
    #   os.makedirs(self.object_dir)
    # if not os.path.exists(self.rlt_dir):
    #   os.makedirs(self.rlt_dir)
    
  def check_Tiling(self, query, key, kmax):
    rlt = []
    for q in query:
      for k in key:
        for m in kmax:
          if q >= k:
            continue
          rlt.append([q, k, m])
    return rlt
  
  def profile(self):
    query = [32, 64, 128]
    key = [32, 64, 128]
    kmax = [64, 128]
    
    rlt = []
    
    template = tuning_fmha_func(self.head_size, self.dtype)
    
    if self.dtype == "float16":
      self.object_dir = self.object_dir + "_fp16"
      self.rlt_dir = self.rlt_dir + "_fp16"
      if self.head_size % 8 != 0:
        self.object_dir = self.object_dir + "_corner"
        
    
    if not os.path.exists(self.object_dir):
      os.makedirs(self.object_dir)
    if not os.path.exists(self.rlt_dir):
      os.makedirs(self.rlt_dir)
    
    self.code_path = self.object_dir + f"/cutlass_flash.cu"
    with open(self.code_path, "w") as f:
      f.write(template)
    
    tiling_setting = self.check_Tiling(query, key, kmax)
    compiler = f"nvcc -O3 {self.code_path} -arch=sm_{self.sm} --std=c++17 -I{self.fmha_cutlass_path}/include -I{self.fmha_cutlass_path}/examples/41_fused_multi_head_attention --expt-relaxed-constexpr"
    
    for setting in tiling_setting:
      object_file = f"{self.object_dir}/fmha_{setting[0]}_{setting[1]}_{setting[2]}"
      rlt.append(object_file)
      
      if os.path.isfile(object_file):
        continue
      object_compile = compiler + f" -DQ_TILE={setting[0]} -DK_TILE={setting[1]} -DKMAX_TILE={setting[2]} -o {object_file}"
      os.system(object_compile)
      
    return rlt
  
  def profile_oracle(self, input_dtype="float16", output_dtype="float16", batch=1, seq_len=512, head_num=12, head_size=64):
    self.batch = int(batch)
    self.seq_len = int(seq_len)
    self.head_num = int(head_num)
    self.head_size = int(head_size)
    self.dtype = str(input_dtype)
    
    if (self.batch, self.seq_len, self.head_num, self.head_size) in self.cache:
      tiling = self.cache[(self.batch, self.seq_len, self.head_num, self.head_size)]
      return tiling
      
    objs = self.profile()
    json_file = self.rlt_dir + f"/{batch}_{seq_len}_{head_num}_{head_size}.json"
    for idx, obj in enumerate(objs):
      run_command = obj + f" -b {batch} -s {seq_len} -n {head_num} -h {head_size} -d {self.rlt_dir}"
      os.system(run_command)
    
    rlt = []
    with open(json_file, "r") as f:
      for line in f:
        json_data = json.loads(line.strip())
        rlt.append(json_data)
    
    sorted_rlt = sorted(rlt, key=lambda x: x["time"])
    
    for idx, value in enumerate(sorted_rlt):
      if value["time"] != -1 and value["time"] != 0:
        fastest_cutlass_time = value["time"]
        fastest_cutlass_tile = value["dim"]
        break
    
    # fastest_cutlass_tile = [64, 64, 64]
    print(f"{self.batch} {self.seq_len} {self.head_num} {self.head_size}")
    print(fastest_cutlass_tile)
    print(fastest_cutlass_time)
    
    self.cache[(self.batch, self.seq_len, self.head_num, self.head_size)] = fastest_cutlass_tile
    
    return fastest_cutlass_tile
    
    
if __name__ == "__main__":
  tmp = ProfileFMHA(path="/home/5862www/curator/LLM/cutlass_rtx4090")
  fastest_tile = tmp.profile_oracle(batch=4, output_dtype="float32", input_dtype="float32")
  
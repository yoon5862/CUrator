import numpy as np
import math
import os

import tvm
from tvm import relay
from tvm import auto_scheduler
from tvm.contrib import graph_executor

from tvm.relay.op.contrib.cutlass import partition_for_cutlass
from tvm.contrib.cutlass_1 import finalize_modules


def cutlass_inference(mod, input, fileName="./cutlass"):
    host = tvm.target.Target("llvm")
    cuda = tvm.target.Target("cuda", host=host)
    dev = tvm.device(str(cuda), 0)
    
    mod = partition_for_cutlass(mod)
    
    cutlass = tvm.target.Target(
        {
            "kind":"cutlass",
            "sm": 80,
            "use_3xtf32": False,
            "split_k_slices": [1],
            "profile_all_alignments": True,
            "find_first_valid": False,
            "use_multiprocessing": True,
            "use_fast_math": True,
            "tmp_dir": fileName,
        },
        host = host,
    )
    
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=[cuda, cutlass], params=input)
    lib = finalize_modules(lib, "compile.so", fileName)
    
    module = graph_executor.GraphModule(lib["default"](dev))
    return module

def tvm_inference(mod, input):
    host = tvm.target.Target("llvm")
    cuda = tvm.target.Target("cuda", host=host)
    dev = tvm.device(str(cuda), 0)
    
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=cuda, params=input)
    module = graph_executor.GraphModule(lib["default"](dev))
    return module

if __name__ =="__main__":
    dim = [(12, 512, 512)]
    batch, M, N = dim[0]
    
    a = relay.var("input", shape=(batch, M, N), dtype="float32")
    
    softmax = relay.nn.softmax(a)
    
    mod = tvm.IRModule.from_expr(softmax)
    mod = relay.transform.InferType()(mod)
    
    a_arr = np.array([i * 0.001 for i in range(batch * M * N)]).reshape(batch, M, N).astype("float32")
    
    params = {}
    
    module = tvm_inference(mod, params)
    module.set_input("input", a_arr)
    module.run()
    
    tvm_output = module.get_output(0).asnumpy()
    
    cutlass_module = cutlass_inference(mod, params)
    cutlass_module.set_input("input", a_arr)
    cutlass_module.run()
    
    cutlass_output = cutlass_module.get_output(0).asnumpy()
    
    cuda = tvm.target.Target("cuda")
    dev = tvm.device(str(cuda), 0)
    
    rlt = cutlass_module.benchmark(dev, number=20, repeat=100)
    print(rlt)
    
    #verification
    tvm_output = tvm_output.reshape(-1)
    cutlass_output = cutlass_output.reshape(-1)
    
    diff = 0
    for value1, value2 in zip(tvm_output, cutlass_output):
        if abs(value1 - value2) > 10e-6:
            diff += 1
    print(f"difference: {diff}")
    print(f"mean: {rlt.mean * 24000}")
    
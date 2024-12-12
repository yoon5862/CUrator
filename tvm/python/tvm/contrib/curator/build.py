import os
import multiprocessing
import tvm
from tvm import relay, runtime
from tvm._ffi.registry import register_func
from tvm.relay.op.contrib.cublas import partition_for_cublas
from tvm.relay.op.contrib.cudnn import partition_for_cudnn
from tvm.relay.op.contrib.curator import partition_for_curator
from tvm.relay.op.contrib.cutlass import partition_for_cutlass

from tvm import auto_scheduler
from tvm.contrib.nvcc import get_cuda_version
from tvm.contrib import graph_executor

from .tune_softmax import ProfileSoftamx
from .tune_fmha import ProfileFMHA
from .gen_gemm import CutlassGemmProfiler
from .rewrite import rewrite_onnx, rewrite_cutlass, rewrite_fmha

host = tvm.target.Target("llvm")
cuda = tvm.target.Target("cuda", host=host)
dev = tvm.device(str(cuda), 0)

device_num = 0
fused_multi_head = False

def ansor_module(mod, params, trial=900, curator_target={}, model="", batch=1, seq_len=512, mixedPrecision=False):
    mod, constant = rewrite_onnx(mod)
    params.update(constant)
    
    dir_name = curator_target["tmp_dir"]
    
    #prepare tvm ansor. split subgraph for tunning
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, cuda)
    
    ansor_dir = f".{dir_name}/ansor_log"
    if not os.path.exists(ansor_dir):
        os.makedirs(ansor_dir)
    
    ansor_log_file = f"{dir_name}/{ansor_dir}/{model}_{batch}_{seq_len}_{mixedPrecision}.json"
    
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10, device=device_num)
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=trial * len(tasks),
            runner=measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(ansor_log_file)],
        )
    
    if not os.path.exists(ansor_log_file):
        tuner.tune(tune_option)
    
    with auto_scheduler.ApplyHistoryBest(ansor_log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=cuda, params=params)
    
    module = graph_executor.GraphModule(lib["default"](dev))
    
    return module
    
def bolt_module(mod, params, sm, tmp_dir):
    mod, constant = rewrite_onnx(mod)
    params.update(constant)
    
    mod = partition_for_cutlass(mod)
    
    cutlass_target = tvm.target.Target(
    {
        "kind": "cutlass",
        "sm": sm,
        "use_3xtf32": False,
        "split_k_slices": [1],
        "profile_all_alignments": True,
        "find_first_valid": False,
        "use_multiprocessing": True,
        "use_fast_math": True,
        "tmp_dir": tmp_dir
    },
    host = host,
    )
    
    
    with tvm.transform.PassContext(opt_level=3):
        graph, mod, tmp_params = tvm.relay.build_module.build(mod, target=[cuda, cutlass_target])
    params.update(tmp_params)
    
    
    lib_path = os.path.join(tmp_dir, "compile.so")
    mod.export_library(lib_path, workspace_dir=tmp_dir, cc="nvcc")
    
    mod = tvm.runtime.load_module(lib_path)
    module = graph_executor.create(graph, mod, dev)
    module.set_input(**params)
    
    return module
    
def cutlass_module_natural(mod, params, target, model="gpt2", moduleTest=""):
    
    mod, constant = rewrite_onnx(mod)
    params.update(constant)
    
    global fused_multi_head
    fused_multi_head = False
    
    mod = partition_for_curator(mod, fmha=False)
    
    mod, workspace = rewrite_cutlass(mod, target)
    params.update(workspace)
    
    curator_target = tvm.target.Target(target, host)
    
    tmp_dir = target["tmp_dir"]
    
    if "gpt2" in model or "bert" in model:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=[cuda, curator_target], params=params)
        lib = finalize_modules(lib, "compile_natural.so", tmp_dir)
        module = graph_executor.GraphModule(lib["default"](dev))
    else:
        with tvm.transform.PassContext(opt_level=3):
            graph, mod, tmp_params = tvm.relay.build_module.build(mod, target=[cuda, curator_target])
        params.update(tmp_params)
        lib_path = os.path.join(tmp_dir, "compile_natural.so")
        mod.export_library(lib_path, workspace_dir=tmp_dir, cc="nvcc")
        
        mod = tvm.runtime.load_module(lib_path)
        module = graph_executor.create(graph, mod, dev)
        module.set_input(**params)
    
    return module

def cutlass_module_fmha(mod, params, target, model="gpt2"):
    
    mod, constant = rewrite_onnx(mod)
    params.update(constant)
    
    global fused_multi_head
    fused_multi_head = True
    
    mod = partition_for_curator(mod, fmha=True)
    
    mod, workspace = rewrite_cutlass(mod, target)
    params.update(workspace)
    curator_target = tvm.target.Target(target, host)

    tmp_dir = target["tmp_dir"]
    
    if "gpt2" in model or "bert" in model:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=[cuda, curator_target], params=params)
        lib = finalize_modules(lib, "compile_fmha.so", tmp_dir)
        module = graph_executor.GraphModule(lib["default"](dev))
    else:
        with tvm.transform.PassContext(opt_level=3):
            graph, mod, tmp_params = tvm.relay.build_module.build(mod, target=[cuda, curator_target])
        params.update(tmp_params)
        lib_path = os.path.join(tmp_dir, "compile_fmha.so")
        mod.export_library(lib_path, workspace_dir=tmp_dir, cc="nvcc")
        
        mod = tvm.runtime.load_module(lib_path)
        module = graph_executor.create(graph, mod, dev)
        module.set_input(**params)
    
    return module

def cutlass_module(mod, params, curator_target):
    print("Natural Module")
    cutlass_natural = cutlass_module_natural(mod, params, curator_target)
    print("Fmha Module")
    cutlass_fmha = cutlass_module_fmha(mod, params, curator_target)
    
    cutlass_natural_rlt = cutlass_natural.benchmark(dev, number=2, repeat=10)
    cutlass_fmha_rlt = cutlass_fmha.benchmark(dev, number=2, repeat=10)
    
    print(f"Natural CUTLASS: {cutlass_natural_rlt.mean * 1000} ms")
    
    return cutlass_fmha if cutlass_natural_rlt.mean > cutlass_fmha_rlt.mean else cutlass_natural

def cublas_module(mod, params):
    mod, constant = rewrite_onnx(mod)
    params.update(constant)
    
    cublas_mod = partition_for_cublas(mod)
    cublas_mod = partition_for_cudnn(cublas_mod)
    
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(cublas_mod, target=[cuda], params=params)
        
    module = graph_executor.GraphModule(lib["default"](dev))
    
    # tmp_dir = "/home/5862www/curator/LLM/cublas_1"
    # with tvm.transform.PassContext(opt_level=3):
    #     graph, mod, tmp_params = tvm.relay.build_module.build(cublas_mod, target=[cuda])
    # params.update(tmp_params)
    
    # lib_path = os.path.join(tmp_dir, "compile.so")
    # mod.export_library(lib_path, workspace_dir=tmp_dir, cc="nvcc")
    
    # mod = tvm.runtime.load_module(lib_path)
    # module = graph_executor.create(graph, mod, dev)
    # module.set_input(**params)
    
    return module

def create_module(mod, params, curator_target):
    # mod, constant = rewrite_onnx(mod)
    # params.update(constant)
    
    cublas = cublas_module(mod, params)
    cutlass = cutlass_module(mod, params, curator_target)
    
    cublas_rlt = cublas.benchmark(dev, number=2, repeat=10)
    cutlass_rlt = cutlass.benchmark(dev, number=2, repeat=10)    
    
    return cutlass if cublas_rlt.mean > cutlass_rlt.mean else cublas

class OpAnnotator(tvm.relay.ExprVisitor):
    """Annotates partitioned functions with shape and dtype information."""

    def __init__(self):
        super().__init__()
        self.signature = {}

    def visit_call(self, call):
        op = call.op
        if isinstance(op, relay.Function) and "Composite" in op.attrs:
            self.signature["op_type"] = op.attrs["Composite"]
            for i, arg in enumerate(op.params):
                self.signature["arg%d_shape" % i] = arg.checked_type.shape
                self.signature["arg%d_dtype" % i] = arg.checked_type.dtype
            self.signature["ret_shape"] = op.ret_type.shape
            self.signature["ret_dtype"] = op.ret_type.dtype
            self.visit(op.body)

        elif isinstance(op, tvm.ir.Op) and op.name in [
            "nn.batch_matmul",
            "nn.conv2d",
            "nn.conv2d_transpose",
            "nn.conv2d_backward_weight",
        ]:
            self.op_attrs = call.attrs
            
        for arg in call.args:
            self.visit(arg)

def handle_batch_matmul(
    cutlass_profiler,
    op_type,
    arg0_shape,
    arg1_shape,
    out_dtype,
    arg0_dtype,
    arg1_dtype,
    transpose_a=False,
    transpose_b=True,
):
    batch = arg0_shape[0]
    if transpose_b == True:
        MM = arg0_shape[1]
        KK = arg0_shape[2]
        NN = arg1_shape[1]
    else:
        MM = arg0_shape[1]
        KK = arg0_shape[2]
        NN = arg1_shape[2]
       
    name, cutlass_op_def, split_k = select_gemm_kernel(
        cutlass_profiler,
        op_type,
        MM,
        KK,
        NN,
        out_dtype,
        arg0_dtype,
        arg1_dtype,
        True,
        batch_count=batch,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
    )
    
    ldb = "K" if transpose_b == True else "N"
    
    return {
        "batch": arg0_shape[0],
        "batch_stride_A": MM * KK,
        "batch_stride_B": KK * NN,
        "batch_stride_C": MM * NN,
        "cutlass_op_def": cutlass_op_def,
        "cutlass_op_name": name,
        "lda": "K",
        "ldb": ldb,
        "ldc": "N",
        "split_k": split_k,
    }

def handle_dense(
    cutlass_profiler,
    op_type,
    arg0_shape,
    arg1_shape,
    out_dtype,
    arg0_dtype,
    arg1_dtype,
):
    MM = arg0_shape[0]
    KK = arg0_shape[1]
    NN = arg1_shape[0]
    
    name, cutlass_op_def, split_k = select_gemm_kernel(
        cutlass_profiler,
        op_type,
        MM,
        KK,
        NN,
        out_dtype,
        arg0_dtype,
        arg1_dtype,
        False,
    )
    
    return {
        "cutlass_op_def": cutlass_op_def,
        "cutlass_op_name": name,
        "lda": "K",
        "ldb": "K",
        "ldc": "N",
        "split_k": split_k
    }

def select_gemm_kernel(
    cutlass_profiler,
    op_type,
    MM,
    KK,
    NN,
    out_dtype,
    arg0_dtype,
    arg1_dtype,
    batched,
    batch_count=1,
    transpose_a=False,
    transpose_b=True,
):
    
    name, cutlass_op_def, split_k = cutlass_profiler.profile(
        op_type,
        MM,
        NN,
        KK,
        out_dtype,
        arg0_dtype,
        arg1_dtype,
        batched=batched,
        batch_count=batch_count,
        transpose_a=transpose_a,
        transpose_b=transpose_b
    )
    
    
    return name, cutlass_op_def, split_k

def tune_cutlass_kernels(mod, tuning_config, tmp_dir=""):
    gemm_profiler = CutlassGemmProfiler(tuning_config)
    softmax_profile = ProfileSoftamx(sm=tuning_config["sm"], path=tmp_dir)
    fmha_profile = ProfileFMHA(sm=tuning_config["sm"], path=tmp_dir)

    for var in mod.get_global_vars():
        fun_name = var.name_hint
        func = mod[fun_name]
        
        if "curator" in fun_name:
            new_func = tune_cutlass_function(func, tuning_config, gemm_profiler, softmax_profile, fmha_profile)
            mod.update_func(var, new_func)

    return mod

def tune_cutlass_function(func, tuning_config, gemm_profiler, softmax_profile, fmha_profile):
    annotator = OpAnnotator()
    annotator.visit(func)
    
    out_shape = annotator.signature["ret_shape"]
    out_dtype = annotator.signature["ret_dtype"]
    op_type = annotator.signature["op_type"]
    
    new_attrs = {"op_type": op_type}
    new_attrs.update(annotator.signature)
    new_attrs.update(func.attrs)
    
    if "fmha" in op_type:
        input_shape = [int(dim) for dim in new_attrs["arg0_shape"]] 
        input_dtype = str(new_attrs["arg0_dtype"])
        
        if "fmha2" in op_type and "gpt" in op_type:
            input_shape = [int(dim) for dim in new_attrs["arg1_shape"]]
            input_dtype = str(new_attrs["arg1_dtype"])
        
        fmha_attrs = select_fmha_kernel(input_shape[0], input_shape[1], input_shape[2], input_shape[3], input_dtype, input_dtype, fmha_profile)
        new_attrs.update(fmha_attrs)
        
        new_attrs = tvm.ir.make_node("DictAttrs", **new_attrs)
        return relay.Function(
            func.params,
            func.body,
            ret_type=func.ret_type,
            type_params=func.type_params,
            attrs=new_attrs
        )
    
    #softmax kernel
    if "softmax_support" in op_type:
        input_shape = new_attrs["arg0_shape"]
        input_dtype = new_attrs["arg0_dtype"]
        
        input_len = len(input_shape)
        batch = input_shape[0] * input_shape[1]
        MM = input_shape[input_len - 2]
        NN = input_shape[input_len - 1]
        
        new_attrs.update(
            select_softmax_kernel(batch, MM, NN, input_dtype, out_dtype, tuning_config["sm"], softmax_profile)
        )
        new_attrs = tvm.ir.make_node("DictAttrs", **new_attrs)
        return relay.Function(
            func.params,
            func.body,
            ret_type=func.ret_type,
            type_params=func.type_params,
            attrs=new_attrs,
        )
    
    arg0_shape = new_attrs["arg0_shape"]
    arg1_shape = new_attrs["arg1_shape"]
    arg0_dtype = new_attrs["arg0_dtype"]
    arg1_dtype = new_attrs["arg1_dtype"]
    
    if "onnx_dense_bias" in str(op_type):
        arg0_shape = new_attrs["arg1_shape"]
        arg1_shape = new_attrs["arg2_shape"]
        arg0_dtype = new_attrs["arg1_dtype"]
        arg1_dtype = new_attrs["arg2_dtype"]
    
    
    if "batch_matmul" in op_type:
        new_attrs["arg0_transpose"] = annotator.op_attrs.transpose_a
        new_attrs["arg1_transpose"] = annotator.op_attrs.transpose_b
        
        transpose_a = annotator.op_attrs.transpose_a
        transpose_b = annotator.op_attrs.transpose_b
        
        new_attrs.update(
            handle_batch_matmul(
                gemm_profiler,
                op_type,
                arg0_shape,
                arg1_shape,
                out_dtype,
                arg0_dtype,
                arg1_dtype,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
            )
        )
        
    elif "dense" in op_type:
        new_attrs.update(
            handle_dense(
                gemm_profiler,
                op_type,
                arg0_shape,
                arg1_shape,
                out_dtype,
                arg0_dtype,
                arg1_dtype,
            )
        )
    else:
        raise ValueError("%s unsupported composite" % op_type)
    
    new_attrs = tvm.ir.make_node("DictAttrs", **new_attrs)
        
    return relay.Function(
        func.params,
        func.body,
        ret_type=func.ret_type,
        type_params=func.type_params,
        attrs=new_attrs,
    )
    
    
def select_softmax_kernel(batch, MM, NN, input_dtype, out_dtype, sm, softmax_profile):
    
    if input_dtype == "float32":
        input_dtype = "float"
    elif input_dtype == "float16":
        input_dtype = "half"
    
    if out_dtype == "float32":
        out_dtype = "float"
    elif out_dtype == "float16":
        out_dtype = "half"
    
    parameter = softmax_profile.profile_oracle(input_type=input_dtype, out_type=out_dtype, batch=int(batch), M=int(MM), N=int(NN))
    
    return {
        "row_per_access": parameter[0],
        "pack_size": parameter[1],
        "col_per_thread": parameter[2],
        "warp_count": parameter[3],
        "cutlass_op_name": "cutlass::support::softmaxWarp"
        }

def select_fmha_kernel(batch, seq_len, head_num, head_size, input_dtype, output_dtype, fmha_profile):
    
    # assert input_dtype != "float32"
    tiling = fmha_profile.profile_oracle(input_dtype=input_dtype, output_dtype=output_dtype, batch=batch, seq_len=seq_len, head_num=head_num, head_size=head_size)
    return {
        "kQueriesPerBlock": tiling[0],
        "kKeysPerBlock": tiling[1],
        "kMaxK": tiling[2],
    }
    
    

def _get_cutlass_path():
    tvm_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../../")
    cutlass_path = os.path.join(tvm_root, "3rdparty/cutlass")
    assert os.path.exists(
        cutlass_path
    ), """The CUTLASS root directory not found in {}.
        Currently, using CUTLASS requires building TVM from source.""".format(
        cutlass_path
    )
    return cutlass_path

def _get_cutlass_compile_options(sm, threads, use_fast_math=False):
    cutlass_root = _get_cutlass_path()
    cutlass_include = os.path.join(cutlass_root, "include")
    cutlass_util_include = os.path.join(cutlass_root, "tools/util/include")
    
    global fused_multi_head
    
    kwargs = {}
    kwargs["cc"] = "nvcc"
    
    if fused_multi_head:
        kwargs["options"] = [
        "-c",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "-gencode=arch=compute_%d,code=[sm_%d,compute_%d]" % (sm, sm, sm),
        "-Xcompiler=-fPIC",
        "-Xcompiler=-Wconversion",
        "-Xcompiler=-fno-strict-aliasing",
        "-O3",
        "-std=c++17",
        "-I" + cutlass_include + "/support/cutlass-3.5.1/include",
        "-I" + cutlass_util_include,
        "-I" + cutlass_include + "/support/cutlass-3.5.1/examples/41_fused_multi_head_attention",
        "-I " + cutlass_include + "/support",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]
    else:
        kwargs["options"] = [
        "-c",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "-gencode=arch=compute_%d,code=[sm_%d,compute_%d]" % (sm, sm, sm),
        "-Xcompiler=-fPIC",
        "-Xcompiler=-Wconversion",
        "-Xcompiler=-fno-strict-aliasing",
        "-O3",
        "-std=c++17",
        "-I" + cutlass_include,
        "-I" + cutlass_util_include,
    ]
        
    if use_fast_math:
        kwargs["options"].append("-DCUTLASS_USE_TANH_FOR_SIGMOID")
    cuda_ver = get_cuda_version()
    if cuda_ver >= (11, 2):
        ncpu = multiprocessing.cpu_count() if threads < 0 else threads
        kwargs["options"].append("-t %d" % ncpu)
    return kwargs

@register_func("contrib.curator.compile")
def compile_cutlass_module(c_source_module, options):
    tmp_dir = options.get("tmp_dir", "./tmp")
    defaults = {"sm": 80, "threads": -1, "use_fast_math": False}
    compile_config = {key: options.get(key, val) for key, val in defaults.items()}
    
    function_names = c_source_module.get_function("get_func_names")()
    compile_options = _get_cutlass_compile_options(**compile_config)
    lib_path = os.path.join(tmp_dir, "cutlass.o")
    
    c_source_module.export_library(lib_path, workspace_dir=tmp_dir, **compile_options)
    return tvm.runtime.load_static_library(lib_path, function_names)


@register_func("relay.ext.curator.compile_for_cutlass")
def compile_for_cutlass(mod, cutlass_target):
    
    assert cutlass_target.kind.name == "curator"
    
    tuning_config = {
        key: cutlass_target.attrs.get(key)
        for key in [
            "sm",
            "tbt_m",
            "tbt_n",
            "tbt_k",
            "pipelining_range",
            "align_range",
            "split_k_range",
            "swizzle_range",
        ]
    }
    
    compile_config = {
        key: cutlass_target.attrs.get(key) for key in ["sm", "use_fast_math", "tmp_dir"]
    }
    tmp_dir = cutlass_target.attrs.get("tmp_dir")
    tuning_config["tmp_dir"] = tmp_dir
    
    # Tune
    mod = tune_cutlass_kernels(mod, tuning_config, tmp_dir)
    
    create_c_source_module = tvm._ffi.get_global_func("relay.ext.curator.create_c_source_module")
    c_module = create_c_source_module(mod)
    
    return compile_cutlass_module(c_module, compile_config)

def finalize_modules(lib, lib_path="compile.so", tmp_dir="./tmp"):
    """Returns lib with any C source, LLVM and static library modules complied and linked in ready
    for use by the graph or AOT executors. This method is not specific to CUTLASS, however it does
    assume nvcc will be used for final compilation and linking. It is provided here for
    convenience.

    Parameters
    ----------
    lib : runtime.Module
        The output from relay.build.

    lib_path : string
        The path to a shared library which will be generated as the result of the build process.

    tmp_dir : string
        A temporary directory where intermediate compiled artifacts will be stored.

    Returns
    -------
    updated_lib : runtime.Module
        The updated library with all compilation and linking completed.

    """
    lib_path = os.path.join(tmp_dir, lib_path)
    lib.export_library(lib_path, workspace_dir=tmp_dir, cc="nvcc")
    return runtime.load_module(lib_path)
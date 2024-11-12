import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprMutator
import numpy as np
from .profile_cutlass_gemm import ProfileBatchedGemm, ProfileGemm
from .profile_cutlass_gemm_tc import ProfileBatchedGemmTC, ProfileGemmTC
from .tune_fmha import ProfileFMHA


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

class OpAnnotator2(tvm.relay.ExprVisitor):
    """Annotates partitioned functions with shape and dtype information."""

    def __init__(self):
        super().__init__()
        self.signature = {}

    def visit_call(self, call):
        op = call.op
        
        if isinstance(op, relay.Function) and "Composite" in op.attrs:
            self.signature["op_type"] = op.attrs["Composite"]
            # print(op.attrs["Composite"])
            # print(op.params)
            
            # if "fmha" in op.attrs["Composite"]:
            #     for i, args in enumerate(op.params):
            #         print(args.checked_type)
                
            #     assert 0 == 100
            
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

#rewrite TVM IR for onnx file        
class ExtractMetaConstants(ExprMutator):
    # Dirty fix for unknown relay.Const[Meta] occurance.
    def __init__(self, mod):
        super().__init__()
        self.constants = {}
        self.params_var = []
        
        self.i = 0
        self.mod = mod

    def visit_constant(self, const: relay.expr.Constant):
        np_data = const.data.numpy()
        # new_const = relay.const(np_data)
        if "relay.Constant" in str(const) and (np_data.shape != ()):
            self.constants[f"input_{self.i}"] = np_data
            new_const = relay.var(f"input_{self.i}", shape=np_data.shape, dtype=str(np_data.dtype))
            self.params_var.append(new_const)
            self.i += 1
            
            return new_const
        else:
            return const     

    def extract_constants(self):
        #change main function's body
        expr = self.visit(self.mod["main"])
        
        func = self.mod["main"]
        params = func.params
        
        #add input parameter into new main function
        for param in params:
            self.params_var.append(param)
        
        #create new main function
        expr = relay.Function(self.params_var, expr.body, expr.ret_type)
        
        #change old main function to new main function
        global_var = self.mod.get_global_vars()
        self.mod.update_func(global_var[0], expr)
        
        return self.mod, self.constants
    
#rewrite TVM IR for FMHA      
class RewriteForFMHA(ExprMutator):
    # Dirty fix for unknown relay.Const[Meta] occurance.
    def __init__(self, mod):
        super().__init__()
        self.constants = {}
        self.params_var = []
        
        self.i = 0
        self.mod = mod
    
    def visit_call(self, call):
        op = call.op
        op_args = [self.visit(arg) for arg in call.args]
        if isinstance(op, tvm.ir.Op):
            op_name = str(op.name)
            
            if "nn.batch_matmul" in op_name:
                # print(f"{op_name}: {len(op_args)}: {len(call.args)}: {call.args[1].op.name}")
                
                is_transpose = call.args[1]
                if "transpose" in is_transpose.op.name:
                    is_reshape = is_transpose.args[0]
                    if "reshape" in is_reshape.op.name:
                        is_transpose2 = is_reshape.args[0]
                        if "transpose" in is_transpose2.op.name:
                            reshape_op = is_transpose2.args[0] # reshape
                            
                            reshape_call = relay.Call(is_reshape.op, [reshape_op], is_reshape.attrs, is_reshape.type_args, is_reshape.span)
                            transpose_call = relay.Call(is_transpose.op, [reshape_call], is_transpose.attrs, is_transpose.type_args, is_transpose.span)
                            op_args[1] = transpose_call
        
                            return relay.Call(call.op, op_args, call.attrs, call.type_args, call.span)
        return super().visit_call(call)
    
    def rewrite(self):
        #change main function's body
        expr = self.visit(self.mod["main"])
        
        func = self.mod["main"]
        params = func.params
        
        #add input parameter into new main function
        for param in params:
            self.params_var.append(param)
        
        #create new main function
        expr = relay.Function(self.params_var, expr.body, expr.ret_type)
        
        #change old main function to new main function
        global_var = self.mod.get_global_vars()
        self.mod.update_func(global_var[0], expr)
        
        return self.mod      
   

def handle_dense(
    op_type,
    arg0_shape,
    arg1_shape,
    arg0_dtype,
    arg1_dtype,
    out_dtype,
    target
):
    MM = arg0_shape[0]
    KK = arg0_shape[1]
    NN = arg1_shape[0]
    
    block_shape, split = select_gemm_kernel(op_type, MM, NN, KK, out_dtype, arg0_dtype, arg1_dtype, target=target)
    
    byte_m = (MM + (block_shape[0] - 1)) // block_shape[0]
    byte_n = (NN + (block_shape[1] - 1)) // block_shape[1]
    
    return byte_m * byte_n, split
    
    
def handle_batch_matmul(
    op_type,
    arg0_shape,
    arg1_shape,
    arg0_dtype,
    arg1_dtype,
    out_dtype,
    transpose_a=False,
    transpose_b=True,
    target={},
):
    # print(f"shape: {arg0_shape}, {arg1_shape}")
    #dimension
    batch = arg0_shape[0]
    if transpose_b == True:
        MM = arg0_shape[1]
        KK = arg0_shape[2]
        NN = arg1_shape[1]
    else:
        MM = arg0_shape[1]
        KK = arg0_shape[2]
        NN = arg1_shape[2]
        
        
    block_shape, split = select_gemm_kernel(op_type, MM, NN, KK, out_dtype, arg0_dtype, arg1_dtype, True, batch, transpose_a, transpose_b, target)
    
    byte_m = (MM + (block_shape[0] - 1)) // block_shape[0]
    byte_n = (NN + (block_shape[1] - 1)) // block_shape[1]
    
    return batch * byte_m * byte_n, split
    

def select_gemm_kernel(
    op_type,
    MM,
    NN,
    KK,
    out_dtype,
    arg0_dtype,
    arg1_dtype,
    batched=False,
    batch_count=1,
    transpose_a=False,
    transpose_b=True,
    target={},
):
    if arg0_dtype == "float32" and out_dtype == "float32":
        if batched == True:
            batchGemmProfile = ProfileBatchedGemm(batch=batch_count, M=MM, N=NN, K=KK, tuning_config=target)
            tile_setting, split_k, _ = batchGemmProfile.eval_cutlassOracle(transpose_a=transpose_a, transpose_b=transpose_b)
        else:
            gemmProfile = ProfileGemm(batch=1, M=MM, N=NN, K=KK, tuning_config=target)
            tile_setting, split_k, _ = gemmProfile.eval_cutlassOracle(transpose_a=transpose_a, transpose_b=transpose_b)
    elif arg0_dtype == "float16" and out_dtype == "float16":
        # print(f"{batch_count}, {MM}, {NN}, {KK}")
        if batched == True:
            batchGemmProfile = ProfileBatchedGemmTC(batch=batch_count, M=MM, N=NN, K=KK, tuning_config=target)
            tile_setting, split_k, _ = batchGemmProfile.eval_cutlassOracle(transpose_a=transpose_a, transpose_b=transpose_b)
        else:
            gemmProfile = ProfileGemmTC(batch=1, M=MM, N=NN, K=KK, tuning_config=target)
            tile_setting, split_k, _ = gemmProfile.eval_cutlassOracle(transpose_a=transpose_a, transpose_b=transpose_b)
    
    block_shape = tile_setting[0]
    
    return block_shape, split_k
    
class ExtractFunction(ExprMutator):
    # Dirty fix for unknown relay.Const[Meta] occurance.
    def __init__(self, mod, target):
        super().__init__()
        self.params = {}
        self.params_var = []
        
        self.i = 0
        self.mod = mod
        self.target = target
        
        self.total = 0
        self.cnt = 0
        
        self.gemm_log = {}
        
        self.fmha_profiler = ProfileFMHA(sm=self.target["sm"], path=self.target["tmp_dir"])
    
    def visit_call(self, call):
        split = 1
        if isinstance(call.op, tvm.ir.expr.GlobalVar):
            new_fn = call.op
            new_args = [self.visit(arg) for arg in call.args]
            
            split_k_func_name = new_fn.name_hint
            split_k_func = self.mod[split_k_func_name]
            
            if "curator" not in split_k_func_name:
                return relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)
            
            if "curator" in split_k_func_name:
                
                annotator = OpAnnotator()
                annotator.visit(split_k_func)
                out_shape = annotator.signature["ret_shape"]
                out_dtype = annotator.signature["ret_dtype"]
                op_type = annotator.signature["op_type"]
                
                new_attrs = {}
                new_attrs = {"op_type": op_type}
                new_attrs.update(annotator.signature)
                new_attrs.update(split_k_func.attrs)
                
                if "softmax" in op_type:
                    return relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)
                
                self.total += 1
                
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
                    
                    if f"{arg0_shape}_{arg1_shape}_{arg0_dtype}_{arg1_dtype}_{transpose_a}_{transpose_b}" in self.gemm_log:
                        split_byte, split = self.gemm_log[f"{arg0_shape}_{arg1_shape}_{arg0_dtype}_{arg1_dtype}_{transpose_a}_{transpose_b}"]
                    else:
                        split_byte, split = handle_batch_matmul(op_type, arg0_shape, arg1_shape, arg0_dtype, arg1_dtype, out_dtype, transpose_a, transpose_b, self.target)
                        self.gemm_log[f"{arg0_shape}_{arg1_shape}_{arg0_dtype}_{arg1_dtype}_{transpose_a}_{transpose_b}"] = (split_byte, split)
                elif "dense" in op_type:
                    if f"{arg0_shape}_{arg1_shape}_{arg0_dtype}_{arg1_dtype}" in self.gemm_log:
                        split_byte, split = self.gemm_log[f"{arg0_shape}_{arg1_shape}_{arg0_dtype}_{arg1_dtype}"]
                    else:
                        split_byte, split = handle_dense(op_type, arg0_shape, arg1_shape, arg0_dtype, arg1_dtype, out_dtype, self.target)
                        self.gemm_log[f"{arg0_shape}_{arg1_shape}_{arg0_dtype}_{arg1_dtype}"] = (split_byte, split)
                elif "fmha2" in op_type:
                    fmha_shape = [int(dim) for dim in new_attrs["arg0_shape"]]
                    
                    if "gpt" in op_type:
                        fmha_shape = [int(dim) for dim in new_attrs["arg1_shape"]]
                    
                    oracle_setting = self.fmha_profiler.profile_oracle(batch=fmha_shape[0], seq_len=fmha_shape[1], head_num=fmha_shape[2], head_size=fmha_shape[3], input_dtype=new_attrs["arg0_dtype"], output_dtype=new_attrs["arg0_dtype"])
                    
                    tranpose_op = new_args[len(new_args) - 1]
                    new_args[len(new_args) - 1] = tranpose_op.args[0]
                    
                    fmha_func_param = [param for param in split_k_func.params]
                    
                    args5_shape = [int(s) for s in new_attrs["arg0_shape"]] if "gpt" not in op_type else [int(s) for s in new_attrs["arg1_shape"]]
                    
                    # args5_shape = [args5_shape[0], args5_shape[2], args5_shape[1], args5_shape[3]]
                    args5_name = str(fmha_func_param[len(fmha_func_param) - 1].name_hint)
                    
                    # args2_shape = [int(s) for s in new_attrs["arg2_shape"]]
                    # args2_shape = [args2_shape[0], args2_shape[3], args2_shape[2], args2_shape[1]]
                    # args2_name = str(fmha_func_param[2].name_hint)
                    
                    # args_dtype = str(new_attrs["arg4_dtype"])
                    
                    if "curator.fmha2_bert_fp32" in op_type:
                        args_dtype = str(new_attrs["arg3_dtype"])
                    if "curator.fmha2_bert_fp16" in op_type:
                        args_dtype = str(new_attrs["arg4_dtype"])
                    if "curator.fmha2_gpt_fp32" in op_type:
                        args_dtype = str(new_attrs["arg3_dtype"])
                    if "curator.fmha2_gpt_fp16" in op_type:
                        args_dtype = str(new_attrs["arg4_dtype"])
                    
                    fmha_func_param[len(new_args) - 1] = relay.var(args5_name, shape=args5_shape, dtype=args_dtype)
                    
                    #plz change split_k_func.body for match type
                    fmha_func_inner = split_k_func.body.op
                    fmha_func_inner_params = [param for param in fmha_func_inner.params]
                    fmha_func_inner_params[len(new_args) - 1] = relay.var(str(fmha_func_inner_params[len(new_args) - 1].name_hint), shape=args5_shape, dtype=args_dtype)
                    # fmha_func_inner_params[2] = relay.var(str(fmha_func_inner_params[2].name_hint), shape=args2_shape, dtype=args_dtype)
                    
                    if oracle_setting[2] > oracle_setting[1]:
                        buffer_size = (fmha_shape[0] * fmha_shape[1] * fmha_shape[2] * fmha_shape[3])
                        fmha_buffer = np.array([0 for _ in range(buffer_size)], dtype=new_attrs["arg3_dtype"])        
                        fmha_var = relay.var(f"fmha_buffer_{self.i}", shape=[buffer_size], dtype=new_attrs["arg3_dtype"])
                        self.params_var.append(fmha_var)
                        self.i += 1
                        new_args.append(fmha_var)
                        
                        fmha_innfer_params = relay.var(f"cutlass_fmha_buffer", shape=[buffer_size], dtype=new_attrs["arg3_dtype"])
                        fmha_func_inner_params.append(fmha_innfer_params)
                        
                        fmha_var2 = relay.var(f"cutlass_fmha_buffer_{self.i}", shape=[buffer_size], dtype=new_attrs["arg3_dtype"])
                        fmha_func_param.append(fmha_var2)
                    
                    inner_func_body = relay.nn.softmax(fmha_func_inner_params[0])
                    
                    if "gpt" in op_type:
                        inner_func_body = relay.nn.softmax(fmha_func_inner_params[1])
                    
                    inner_func = relay.Function(fmha_func_inner_params, inner_func_body, fmha_func_inner.ret_type, type_params=fmha_func_inner.type_params, attrs=fmha_func_inner.attrs, span=fmha_func_inner.span)
                    inner_call = relay.Call(inner_func, fmha_func_param)
                    
                    # create fmha attention function
                    new_func = relay.Function(fmha_func_param, inner_call, split_k_func.ret_type, type_params=split_k_func.type_params, attrs=split_k_func.attrs, span=split_k_func.span)
                    self.mod.update_func(new_fn, new_func)
                    
                    return relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)
                
                elif "fmha_openllama" in op_type:
                    # assert new_attrs["arg0_dtype"] == "float16" and new_attrs["arg2_dtype"] == "float16" and new_attrs["arg5_dtype"] == "float16"
                    
                    fmha_shape = [int(dim) for dim in new_attrs["arg0_shape"]]
                    oracle_setting = self.fmha_profiler.profile_oracle(batch=fmha_shape[0], seq_len=fmha_shape[2], head_num=fmha_shape[1], head_size=fmha_shape[3], input_dtype=new_attrs["arg5_dtype"], output_dtype=new_attrs["arg5_dtype"])
                    
                    
                    #check last for llama
                    # tranpose_op = new_args[len(new_args) - 1].args[0]
                    # new_args[len(new_args) - 1] = tranpose_op
                    reshape_op = new_args[len(new_args) - 1]
                    new_args[len(new_args) - 1] = relay.transpose(reshape_op, axes=[0, 2, 1, 3])
                    
                    
                    
                    tranpose_op = new_args[2]
                    new_args[2] = relay.transpose(tranpose_op, axes=[0, 2, 1, 3])
                    
                    scale_add = new_args[0]
                    new_args[0] = relay.transpose(scale_add, axes=[0, 2, 1, 3])
                    
                    
                    fmha_func_param = [param for param in split_k_func.params]
                    
                    args5_shape = [int(s) for s in new_attrs["arg5_shape"]]
                    args5_shape = [args5_shape[0], args5_shape[2], args5_shape[1], args5_shape[3]]
                    args5_name = str(fmha_func_param[len(fmha_func_param) - 1].name_hint)
                    
                    args2_shape = [int(s) for s in new_attrs["arg2_shape"]]
                    args2_shape = [args2_shape[0], args2_shape[2], args2_shape[1], args2_shape[3]]
                    args2_name = str(fmha_func_param[2].name_hint)
                    
                    args0_shape = [int(s) for s in new_attrs["arg0_shape"]]
                    args0_shape = [args0_shape[0], args0_shape[2], args0_shape[1], args0_shape[3]]
                    args0_name = str(fmha_func_param[0].name_hint)
                    
                    args_dtype = str(new_attrs["arg5_dtype"])
                    
                    fmha_func_param[len(new_args) - 1] = relay.var(args5_name, shape=args5_shape, dtype=args_dtype)
                    fmha_func_param[2] = relay.var(args2_name, shape=args2_shape, dtype=args_dtype)
                    fmha_func_param[0] = relay.var(args0_name, shape=args0_shape, dtype=args_dtype)
                    
                    #plz change split_k_func.body for match type
                    fmha_func_inner = split_k_func.body.op
                    fmha_func_inner_params = [param for param in fmha_func_inner.params]
                    fmha_func_inner_params[len(new_args) - 1] = relay.var(str(fmha_func_inner_params[len(new_args) - 1].name_hint), shape=args5_shape, dtype=args_dtype)
                    fmha_func_inner_params[2] = relay.var(str(fmha_func_inner_params[2].name_hint), shape=args2_shape, dtype=args_dtype)
                    fmha_func_inner_params[0] = relay.var(str(fmha_func_inner_params[0].name_hint), shape=args0_shape, dtype=args_dtype)
                    
                    
                    if oracle_setting[2] > oracle_setting[1]:
                        buffer_size = (fmha_shape[0] * fmha_shape[1] * fmha_shape[2] * fmha_shape[3])
                        # buffer_size = fmha_shape[0] * fmha_shape[2] * fmha_shape[3]
                        fmha_buffer = np.array([0 for _ in range(buffer_size)], dtype=new_attrs["arg0_dtype"])        
                        fmha_var = relay.var(f"fmha_buffer_{self.i}", shape=[buffer_size], dtype=new_attrs["arg0_dtype"])
                        self.params_var.append(fmha_var)
                        self.params[f"split_buffer_{self.i}"] = fmha_buffer
                        self.i += 1
                        new_args.append(fmha_var)
                        
                        fmha_innfer_params = relay.var(f"cutlass_fmha_buffer", shape=[buffer_size], dtype=new_attrs["arg0_dtype"])
                        fmha_func_inner_params.append(fmha_innfer_params)
                        
                        fmha_var2 = relay.var(f"cutlass_fmha_buffer_{self.i}", shape=[buffer_size], dtype=new_attrs["arg0_dtype"])
                        fmha_func_param.append(fmha_var2)
                        
                        # # flash attention splitk
                        # flash_split = 1
                        
                        # #first
                        # buffer_size = flash_split * fmha_shape[0] * fmha_shape[2] * fmha_shape[3]
                        # fmha_buffer = np.array([0 for _ in range(buffer_size)], dtype="float32")        
                        # fmha_var = relay.var(f"fmha_buffer_{self.i}", shape=[buffer_size], dtype="float32")
                        # self.params_var.append(fmha_var)
                        # self.params[f"split_buffer_{self.i}"] = fmha_buffer
                        # self.i += 1
                        # new_args.append(fmha_var)
                        
                        # fmha_innfer_params = relay.var(f"softmax_lse", shape=[buffer_size], dtype="float32")
                        # fmha_func_inner_params.append(fmha_innfer_params)
                        
                        # fmha_var2 = relay.var(f"cutlass_fmha_buffer_{self.i}", shape=[buffer_size], dtype="float32")
                        # fmha_func_param.append(fmha_var2)
                        
                        # # second
                        # buffer_size = flash_split * fmha_shape[0] * fmha_shape[2] * fmha_shape[3] * fmha_shape[3]
                        # fmha_buffer = np.array([0 for _ in range(buffer_size)], dtype="float32")        
                        # fmha_var = relay.var(f"fmha_buffer_{self.i}", shape=[buffer_size], dtype="float32")
                        # self.params_var.append(fmha_var)
                        # self.params[f"split_buffer_{self.i}"] = fmha_buffer
                        # self.i += 1
                        # new_args.append(fmha_var)
                        
                        # fmha_innfer_params = relay.var(f"out_accum", shape=[buffer_size], dtype="float32")
                        # fmha_func_inner_params.append(fmha_innfer_params)
                        
                        # fmha_var2 = relay.var(f"cutlass_fmha_buffer_{self.i}", shape=[buffer_size], dtype="float32")
                        # fmha_func_param.append(fmha_var2)
                        
                        # #flash attention splitk
                    
                    inner_func_body = relay.nn.softmax(fmha_func_inner_params[0])
                    
                    inner_func = relay.Function(fmha_func_inner_params, inner_func_body, fmha_func_inner.ret_type, type_params=fmha_func_inner.type_params, attrs=fmha_func_inner.attrs, span=fmha_func_inner.span)
                    inner_call = relay.Call(inner_func, fmha_func_param)
                    
                    new_func = relay.Function(fmha_func_param, inner_call, split_k_func.ret_type, type_params=split_k_func.type_params, attrs=split_k_func.attrs, span=split_k_func.span)
                    self.mod.update_func(new_fn, new_func)
                    
                    return relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)
                
                elif "fmha" in op_type:
                    # profile fmha attention for split k algorithm
                    # assert new_attrs["arg0_dtype"] == "float16" and new_attrs["arg2_dtype"] == "float16" and new_attrs["arg5_dtype"] == "float16"
                    
                    fmha_shape = [int(dim) for dim in new_attrs["arg0_shape"]]
                    oracle_setting = self.fmha_profiler.profile_oracle(batch=fmha_shape[0], seq_len=fmha_shape[1], head_num=fmha_shape[2], head_size=fmha_shape[3], input_dtype=new_attrs["arg5_dtype"], output_dtype=new_attrs["arg5_dtype"])
                    # oracle_setting = [32, 64, 128]
                    
                    tranpose_op = new_args[len(new_args) - 1]
                    new_args[len(new_args) - 1] = tranpose_op.args[0]
                    
                    # reshape_op = new_args[2]
                    # new_transpose_op = relay.transpose(reshape_op, axes=[0, 3, 2, 1])
                    # new_args[2] = new_transpose_op
                    
                    fmha_func_param = [param for param in split_k_func.params]
                    
                    args5_shape = [int(s) for s in new_attrs["arg5_shape"]]
                    args5_shape = [args5_shape[0], args5_shape[2], args5_shape[1], args5_shape[3]]
                    args5_name = str(fmha_func_param[len(fmha_func_param) - 1].name_hint)
                    
                    args2_shape = [int(s) for s in new_attrs["arg2_shape"]]
                    args2_shape = [args2_shape[0], args2_shape[3], args2_shape[2], args2_shape[1]]
                    args2_name = str(fmha_func_param[2].name_hint)
                    
                    args_dtype = str(new_attrs["arg5_dtype"])
                    
                    fmha_func_param[len(new_args) - 1] = relay.var(args5_name, shape=args5_shape, dtype=args_dtype)
                    # fmha_func_param[2] = relay.var(args2_name, shape=args2_shape, dtype=args_dtype)
                    
                    #plz change split_k_func.body for match type
                    fmha_func_inner = split_k_func.body.op
                    fmha_func_inner_params = [param for param in fmha_func_inner.params]
                    fmha_func_inner_params[len(new_args) - 1] = relay.var(str(fmha_func_inner_params[len(new_args) - 1].name_hint), shape=args5_shape, dtype=args_dtype)
                    # fmha_func_inner_params[2] = relay.var(str(fmha_func_inner_params[2].name_hint), shape=args2_shape, dtype=args_dtype)
                    
                    if oracle_setting[2] > oracle_setting[1]:
                        buffer_size = (fmha_shape[0] * fmha_shape[1] * fmha_shape[2] * fmha_shape[3])
                        fmha_buffer = np.array([0 for _ in range(buffer_size)], dtype=new_attrs["arg0_dtype"])        
                        fmha_var = relay.var(f"fmha_buffer_{self.i}", shape=[buffer_size], dtype=new_attrs["arg0_dtype"])
                        self.params_var.append(fmha_var)
                        self.i += 1
                        new_args.append(fmha_var)
                        
                        fmha_innfer_params = relay.var(f"cutlass_fmha_buffer", shape=[buffer_size], dtype=new_attrs["arg0_dtype"])
                        fmha_func_inner_params.append(fmha_innfer_params)
                        
                        fmha_var2 = relay.var(f"cutlass_fmha_buffer_{self.i}", shape=[buffer_size], dtype=new_attrs["arg0_dtype"])
                        fmha_func_param.append(fmha_var2)
                    
                    inner_func_body = relay.nn.softmax(fmha_func_inner_params[0])
                    
                    inner_func = relay.Function(fmha_func_inner_params, inner_func_body, fmha_func_inner.ret_type, type_params=fmha_func_inner.type_params, attrs=fmha_func_inner.attrs, span=fmha_func_inner.span)
                    inner_call = relay.Call(inner_func, fmha_func_param)
                    
                    # create fmha attention function
                    new_func = relay.Function(fmha_func_param, inner_call, split_k_func.ret_type, type_params=split_k_func.type_params, attrs=split_k_func.attrs, span=split_k_func.span)
                    self.mod.update_func(new_fn, new_func)
                    
                    return relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)
                
                # print(f"{split_k_func_name}, {arg0_shape}, {arg1_shape}, {out_shape} -> {split}")
                if int(split) == 1:
                    return relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)
                else:
                    self.cnt += 1
            
            #new params for split-k buffer
            split_k_buffer = np.array([0 for _ in range(int(split_byte))], dtype="int32")
            self.params[f"split_buffer_{self.i}"] = split_k_buffer
            
            var = relay.var(f"split_buffer_{self.i}", shape=split_k_buffer.shape, dtype=str(split_k_buffer.dtype))
            self.params_var.append(var)
            self.i += 1
            
            #get split k memory
            new_args.append(var)
            
            var2 = relay.var(f"cutlass_split_buffer_{self.i}", shape=split_k_buffer.shape, dtype=str(split_k_buffer.dtype))
            new_split_k_params = [params for params in split_k_func.params]
            new_split_k_params.append(var2)
            
            #change wrapper function
            sub_func_body = split_k_func.body.op.body
            sub_func_params = [params for params in split_k_func.body.op.params]
            sub_func_var = relay.var(f"cutlass_split_buffer", shape=split_k_buffer.shape, dtype=str(split_k_buffer.dtype))
            sub_func_params.append(sub_func_var)
            inner_func = relay.Function(sub_func_params, sub_func_body, attrs=split_k_func.body.op.attrs)
            inner_call = relay.Call(inner_func, new_split_k_params)
            
            new_split_k_func = relay.Function(new_split_k_params, inner_call, split_k_func.ret_type, attrs=split_k_func.attrs)
            self.mod.update_func(new_fn, new_split_k_func)
            
            return relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)
        
        return super().visit_call(call)
    
    def rewrite_func(self):
        #change main function's body, params
        new_main_body = self.visit(self.mod["main"])
        new_main_params = [params for params in new_main_body.params]
        for params in self.params_var:
            new_main_params.append(params)
        
        new_main = relay.Function(new_main_params, new_main_body.body, new_main_body.ret_type)
        for var in self.mod.get_global_vars():
            if str(var.name_hint) == "main":
                self.mod.update_func(var, new_main)
                break
        
        return self.mod, self.params

#rewrite onnx ir    
#because onnx's parameter is constant
#cublas, cutlass boyc doesn't work well with constant parameters
def rewrite_onnx(mod):
    num = ExtractMetaConstants(mod)
    new_mod, params = num.extract_constants()
    return new_mod, params

def rewrite_fmha(mod):
    rewrite_fmha = RewriteForFMHA(mod)
    new_mod = rewrite_fmha.rewrite()
    return new_mod

#rewrite cutlass
#split-k must be allocate in prepare
def rewrite_cutlass(mod, target):
    num = ExtractFunction(mod, target)
    new_mod, add_params = num.rewrite_func()
        
    return new_mod, add_params
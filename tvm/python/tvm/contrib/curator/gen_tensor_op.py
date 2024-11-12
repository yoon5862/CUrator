import os
import re
import tempfile
import subprocess
import multiprocessing

import tvm._ffi
from tvm.tir import IntImm
from tvm.runtime import Object
from . import _ffi_api as ffi
from .library import (
    MathInstruction,
    DataType,
    DataTypeTag,
    OpcodeClass,
    MathOperation,
    TileDescription,
    EpilogueFunctor,
)

from .tune_softmax import instantiate_softmax_template
from .gemm_operation import instantiate_gemm_template
from .tune_fmha import instantiate_fmha_template

dtype_map = {
    "int8": DataType.s8,
    "uint8": DataType.u8,
    "int32": DataType.s32,
    "float32": DataType.f32,
    "float16": DataType.f16,
}

# (Epilogue functor name, no_beta_scaling)
EPILOGUE_MAP = {
    "curator.dense": (EpilogueFunctor.LinearCombination, False),
    "curator.dense_bias": (EpilogueFunctor.LinearCombinationBias, True),
    "curator.onnx_dense_bias": (EpilogueFunctor.LinearCombinationBias, True),
    "curator.dense_bias_relu": (EpilogueFunctor.LinearCombinationRelu, True),
    "curator.onnx_dense_bias_relu": (EpilogueFunctor.LinearCombinationRelu, True),
    "curator.dense_bias_gelu_fp16": (EpilogueFunctor.LinearCombinationGelu, False),
    "curator.onnx_dense_bias_gelu_fp16": (EpilogueFunctor.LinearCombinationGelu, False),
    "curator.dense_bias_gelu_fp32": (EpilogueFunctor.LinearCombinationGelu, False),
    "curator.onnx_dense_bias_gelu_fp32": (EpilogueFunctor.LinearCombinationGelu, False),
    "curator.batch_matmul": (EpilogueFunctor.LinearCombination, False),
    "curator.batch_matmul_bias": (EpilogueFunctor.LinearCombinationBias, True),
    "curator.batch_matmul_bias_relu": (EpilogueFunctor.LinearCombinationRelu, True),
    "curator.batch_matmul_bias_gelu_fp32": (EpilogueFunctor.LinearCombinationGelu, False),
    "curator.batch_matmul_bias_gelu_fp16": (EpilogueFunctor.LinearCombinationGelu, False),
}

class CodegenResult(Object):
    """The holder for the generated code and required headers."""

    def __init__(self, code, headers):
        self.__init_handle_by_constructor__(ffi.CodegenResult, code, headers)

@tvm._ffi.register_func("contrib.curator.instantiate_template")
def instantiate_template(func_name, annotations, func_args):
    attrs = {}
    
    if "fmha" in func_name:
        # attrs["qkv_shape"] = annotations["arg0_shape"]
        # attrs["ElementInputA"] = DataTypeTag[dtype_map[annotations["arg0_dtype"]]]
        
        if "gpt" in func_name and "fmha2" in func_name:
            attrs["qkv_shape"] = annotations["arg1_shape"]
            attrs["ElementInputA"] = DataTypeTag[dtype_map[annotations["arg1_dtype"]]]
        else:
            attrs["qkv_shape"] = annotations["arg0_shape"]
            attrs["ElementInputA"] = DataTypeTag[dtype_map[annotations["arg0_dtype"]]]
        
        for k in ["kQueriesPerBlock", "kKeysPerBlock", "kMaxK"]:
            if k in annotations:
                attrs[k] = str(int(annotations[k]))
        
        torch_var1 = True if "fmha2" in func_name else False
        
        code = instantiate_fmha_template(attrs, func_args, torch_var1, func_name)
        return CodegenResult(code, ["kernel_forward.h",]) #"kernel_forward.h", "flash.h", "static_switch.h"
    
    if "softmax" in func_name:
        arg0_shape = annotations["arg0_shape"]
        len_arg0 = len(arg0_shape)
        
        attrs["M"] = str(int(arg0_shape[len_arg0 - 2]))
        attrs["N"] = str(int(arg0_shape[len_arg0 - 1]))
        
        if len_arg0 == 4:
            attrs["batch"] = str(int(arg0_shape[0]) * int(arg0_shape[1]))
        elif len_arg0 == 3:
            attrs["batch"] = str(int(arg0_shape[0]))
        elif len_arg0 == 2:
            attrs["batch"] = str(1)
        
        for k in ["row_per_access", "pack_size", "col_per_thread", "warp_count",]:
            if k in annotations:
                attrs[k] = str(int(annotations[k]))
        
        attrs["cutlass_op_name"] = annotations["cutlass_op_name"]
        #insert code here!
        attrs["ElementInputA"] = DataTypeTag[dtype_map[annotations["arg0_dtype"]]]
        attrs["ElementOutput"] = DataTypeTag[dtype_map[annotations["ret_dtype"]]]
        assert attrs["ElementInputA"] == attrs["ElementOutput"], "Input Dtype and output Dtype must be same"
        code = instantiate_softmax_template(attrs, func_args)
        
        return CodegenResult(code, ["support/softmax.cuh"])
    
    
    for k in ["lda", "ldb", "ldc", "cutlass_op_def", "cutlass_op_name", "op_type",]:
        if k in annotations:
            attrs[k] = annotations[k]

    arg0_shape = annotations["arg0_shape"]
    arg1_shape = annotations["arg1_shape"]
    
    if "arg2_shape" in annotations:
        attrs["arg2_shape"] = list(annotations["arg2_shape"])
    
    
    attrs["ElementInputA"] = DataTypeTag[dtype_map[annotations["arg0_dtype"]]]
    attrs["ElementInputB"] = DataTypeTag[dtype_map[annotations["arg1_dtype"]]]
    
    attrs["ElementOutput"] = DataTypeTag[dtype_map[annotations["ret_dtype"]]]
    attrs["split_k"] = str(annotations["split_k"])
    
    if "curator.onnx_dense_bias" in str(func_name):
        tmp_func_args = [func_args[1], func_args[2], func_args[0]]
        if len(func_args) == 4:
            tmp_func_args.append(func_args[3])
        func_args = tmp_func_args
        
        arg2_shape = annotations["arg0_shape"]
        arg0_shape = annotations["arg1_shape"]
        arg1_shape = annotations["arg2_shape"]
        
        arg0_dtype = annotations["arg1_dtype"]
        arg1_dtype = annotations["arg2_dtype"]
        
        attrs["ElementInputA"] = DataTypeTag[dtype_map[arg0_dtype]]
        attrs["ElementInputB"] = DataTypeTag[dtype_map[arg1_dtype]]
           
        attrs["arg2_shape"] = list(arg2_shape)

    headers = []
    
    if "relu" in func_name:
        headers.append("cutlass/epilogue/thread/linear_combination_bias_relu.h")
    elif "gelu" in func_name:
        headers.append("cutlass/epilogue/thread/linear_combination_gelu.h")
    elif "sigmoid" in func_name:
        headers.append("cutlass/epilogue/thread/linear_combination_sigmoid.h")
    elif "silu" in func_name:
        headers.append("cutlass/epilogue/thread/linear_combination_silu.h")
    elif "hardswish" in func_name:
        headers.append("cutlass/epilogue/thread/linear_combination_hardswish.h")
    else:
        headers.append("cutlass/epilogue/thread/linear_combination.h")

    if "residual" in func_name:
        headers.append("cutlass/epilogue/thread/linear_combination_residual_block.h")

    def get_dim(shape_annot, var_name, axis_idx, batched_offset=0):
        if isinstance(shape_annot, IntImm):
            return str(int(shape_annot))
        return "{}->shape[{}]".format(var_name, batched_offset + axis_idx)

    def get_batch_stride(stride_annot, arg0_idx, arg1_idx, arg0_axis_idx, arg1_axis_idx):
        if isinstance(stride_annot, IntImm):
            return str(int(stride_annot))
        dim1 = func_args[arg0_idx] + "->shape[{}]".format(arg0_axis_idx)
        dim2 = func_args[arg1_idx] + "->shape[{}]".format(arg1_axis_idx)
        return dim1 + " * " + dim2

    if "dense" in func_name or "matmul" in func_name:
        batched = "batch_matmul" in func_name
        batched_offset = 1 if batched else 0
        
        #lhs matrix layout is always RowMajor
        attrs["K"] = str(int(arg0_shape[batched_offset + 1]))
        if batched and len(arg0_shape) == 4:
            attrs["K"] = str(int(arg0_shape[batched_offset + 2]))
        
        attrs["M"] = get_dim(arg0_shape[batched_offset], func_args[0], 0, batched_offset)
        if batched and len(arg0_shape) == 4:
            attrs["M"] = get_dim(arg0_shape[batched_offset + 1], func_args[0], 0, batched_offset)
            
        #matrix layout can be RowMajor and ColumnMajor 
        if annotations["ldb"] == "N":
            if len(arg1_shape) == 4 and batched:
                attrs["N"] = get_dim(arg1_shape[batched_offset + 2], func_args[1], 1, batched_offset)
            elif len(arg1_shape) == 3 and batched:
                attrs["N"] = get_dim(arg1_shape[batched_offset + 1], func_args[1], 1, batched_offset)
            elif len(arg1_shape) == 2 and batched:
                attrs["N"] = get_dim(arg1_shape[batched_offset], func_args[1], 1, batched_offset)
            else:
                attrs["N"] = get_dim(arg1_shape[batched_offset + 1], func_args[1], 1, batched_offset)
        else:
            if len(arg1_shape) == 4 and batched:
                attrs["N"] = get_dim(arg1_shape[batched_offset + 1], func_args[1], 0, batched_offset)
            elif len(arg1_shape) == 3 and batched:
                attrs["N"] = get_dim(arg1_shape[batched_offset], func_args[1], 0, batched_offset)
            elif len(arg1_shape) == 2 and batched:
                attrs["N"] = get_dim(arg1_shape[0], func_args[1], 0, batched_offset)
            else:
                attrs["N"] = get_dim(arg1_shape[batched_offset], func_args[1], 0, batched_offset)

        if batched:
            headers.append("cutlass/gemm/device/gemm_batched_splitK.h")
            attrs["batch"] = get_dim(arg0_shape[0], func_args[0], 0)
            
            if batched and len(arg0_shape) == 4:
                attrs["batch"] = str(int(arg0_shape[0]) * int(arg0_shape[1]))
            
            attrs["batch_stride_A"] = get_batch_stride(annotations["batch_stride_A"], 0, 0, 1, 2)
            attrs["batch_stride_B"] = get_batch_stride(annotations["batch_stride_B"], 1, 1, 1, 2)

            if annotations["ldb"] == "N":
                attrs["batch_stride_C"] = get_batch_stride(
                    annotations["batch_stride_C"], 0, 1, 1, 2
                )
            else:
                attrs["batch_stride_C"] = get_batch_stride(
                    annotations["batch_stride_C"], 0, 1, 1, 1
                )
        else:
            headers.append("cutlass/gemm/device/gemm.h")
            
        code = instantiate_gemm_template(attrs, func_args)
        return CodegenResult(code, headers)
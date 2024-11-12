# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name
"""Patterns supported CUTLASS."""
from functools import partial

from tvm import relay
from tvm.ir.transform import PassContext, Sequential
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.op.contrib.register import register_pattern_table  # type: ignore

from ...dataflow_pattern import is_constant, is_op, wildcard, is_tuple



def make_gelu_pattern(bias_out, out_dtype="float16"):
    mul = is_op("multiply")(bias_out, is_constant() | wildcard())
    if out_dtype == "float16":
        erf = is_op("cast")(is_op("erf")(is_op("cast")(mul)))
    else:
        erf = is_op("erf")(mul)
    mul_half = is_op("multiply")(erf, is_constant() | wildcard())
    add = is_op("add")(mul_half, is_constant() | wildcard())
    return is_op("multiply")(add, bias_out)

def make_gelu_pattern_pytorch(bias_out, out_dtype="float16"):
    mul = is_op("multiply")(bias_out, is_constant() | wildcard())
    if out_dtype == "float16":
        erf = is_op("cast")(is_op("erf")(is_op("cast")(mul)))
    else:
        erf = is_op("erf")(mul)
    mul_half = is_op("multiply")(erf, is_constant() | wildcard())
    add = is_op("add")(is_constant() | wildcard(), mul_half)
    return is_op("multiply")(bias_out, add)

def make_gelu_pattern_onnx(bias_out, out_dtype="float16"):
    divide = is_op("divide")(bias_out, is_constant() | wildcard())
    if out_dtype == "float16":
        erf = is_op("cast")(is_op("erf")(divide))
    else:
        erf = is_op("erf")(divide)
    
    add = is_op("add")(erf, is_constant() | wildcard())
    multiply = is_op("multiply")(bias_out, add)
    return is_op("multiply")(multiply, is_constant() | wildcard())
    
def make_gemm_pattern(with_bias=True, with_act=None, out_dtype="float16"):
    """Create a pattern for dense op followed by activations."""
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    gemm = is_op("nn.dense")(data, weight)
    if with_bias:
        add_or_bias_add = is_op("add") | is_op("nn.bias_add")
        gemm_out = add_or_bias_add(gemm, bias)
    else:
        gemm_out = gemm

    if with_act is None:
        return gemm_out
    if isinstance(with_act, str) and with_act == "relu":
        return is_op("nn.relu")(gemm_out)

    assert isinstance(with_act, str) and with_act == "gelu"
    return make_gelu_pattern(gemm_out, out_dtype)

def make_gemm_pattern_onnx(with_bias=True, with_act=None, out_dtype="float16"):
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    
    gemm = is_op("nn.dense")(data, weight)
    if with_bias:
        gemm = is_op("reshape")(gemm)
        add_ro_bias_add = is_op("add") | is_op("nn.bias_add")
        gemm_out = add_ro_bias_add(bias, gemm)
    else:
        gemm_out = gemm
    
    if with_act is None:
        return gemm_out
    if isinstance(with_act, str) and with_act == "relu":
        return is_op("nn.relu")(gemm_out)
    assert isinstance(with_act, str) and with_act == "gelu"
    return make_gelu_pattern_onnx(gemm_out, out_dtype)

def make_batch_matmul_pattern_custom(with_bias=True, with_act=None, out_dtype="float32"):
    data = wildcard()
    weight = wildcard()
    bias = wildcard()

    batch_gemm = is_op("nn.batch_matmul")(data, weight)
    
    if with_bias:
        add_or_bias_add = is_op("add") | is_op("nn.bias_add")
        batch_gemm_out = add_or_bias_add(batch_gemm, bias)
    else:
        batch_gemm_out = batch_gemm
    
    if with_act is None:
        return batch_gemm_out
    if isinstance(with_act, str) and with_act == "relu":
        return is_op("nn.relu")(batch_gemm_out)
    
    assert isinstance(with_act, str) and with_act == "gelu"
    return make_gelu_pattern(batch_gemm_out, out_dtype)

def make_batch_matmul_pattern_pytorch(with_bias=False, with_act=None, transpose=False, out_dtype="float32"):
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    
    broadcast_data = is_op("broadcast_to")(data)
    reshape_data = is_op("reshape")(broadcast_data)
    
    if transpose == True:
        transpose_weight = is_op("transpose")(weight)
    else:
        transpose_weight = weight
    
    broadcast_weight = is_op("broadcast_to")(transpose_weight)
    reshape_weight = is_op("reshape")(broadcast_weight)    
    
    batch_gemm = is_op("nn.batch_matmul")(reshape_data, reshape_weight)
    
    if with_bias:
        bias_reshape = is_op("reshape")(batch_gemm)
        bias_squeeze = is_op("squeeze")(bias_reshape)
        out_batch_gemm = is_op("nn.bias_add")(bias_squeeze, bias)
    else:
        out_batch_gemm = batch_gemm
    
    if with_act is None:
        return out_batch_gemm
    if isinstance(with_act, str) and with_act == "relu":
        return is_op("nn.relu")(out_batch_gemm)
    
    assert isinstance(with_act, str) and with_act == "gelu"
    return make_gelu_pattern_pytorch(out_batch_gemm, out_dtype)

def make_batch_matmul_pattern():
    return is_op("nn.batch_matmul")(wildcard(), wildcard())

def make_conv2d_pattern(with_bias=False, with_act=None):
    """Create a pattern for dense op followed by activations."""
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    conv2d = is_op("nn.conv2d")(data, weight)
    if with_bias:
        add_or_bias_add = is_op("add") | is_op("nn.bias_add")
        conv2d_out = add_or_bias_add(conv2d, bias)
    else:
        conv2d_out = conv2d

    if with_act is not None:
        if with_act == "relu":
            return is_op("nn.relu")(conv2d_out)
        if with_act == "sigmoid":
            return is_op("sigmoid")(conv2d_out)
        if with_act == "silu":
            return is_op("multiply")(conv2d_out, is_op("sigmoid")(conv2d_out))
        if with_act == "hardswish":
            rhs = is_op("divide")(
                is_op("clip")(is_op("add")(conv2d_out, is_constant())), is_constant()
            )
            return is_op("multiply")(conv2d_out, rhs)

        raise ValueError("Unknown activation %s." % with_act)

    return conv2d_out

def make_conv2d_transpose_pattern():
    return is_op("nn.conv2d_transpose")(wildcard(), wildcard())

def make_conv2d_backward_weight_pattern():
    return is_op("nn.conv2d_backward_weight")(wildcard(), wildcard())

def make_residual_block_pattern(tensor_op_out, binary_op="add", with_act="relu"):
    """Add pattern for residual blocks."""
    residual_input = wildcard()
    binary_out = is_op(binary_op)(tensor_op_out, residual_input) | is_op(binary_op)(
        residual_input, tensor_op_out
    )

    if with_act is not None and with_act == "relu":
        return is_op("nn.relu")(binary_out)

    return binary_out

def check_dtype(lhs, rhs):
    """Check if dtypes in the given workload are supported by CUTLASS."""
    return (
        (lhs.dtype == "float16" and rhs.dtype == "float16")
        or (lhs.dtype == "float32" and rhs.dtype == "float32")
        or (lhs.dtype in ["int8", "uint8"] and rhs.dtype in ["int8", "uint8"])
    )

def get_root_call(call, root_op_name):
    
    if not isinstance(call, relay.Call):
        return None
    if str(call.op.name) == root_op_name:
        return call
    return get_root_call(call.args[0], root_op_name)

def get_transpose(call, root_op_name):
    call = call.args[0] #broadcast
    transpose = call.args[0]
    return list(transpose.attrs.axes)

def check_gemm(call):
    """Check if the given dense workload can be offloaded to CUTLASS."""
    dense = get_root_call(call, "nn.dense")
    lhs = dense.args[0].checked_type
    rhs = dense.args[1].checked_type
    return check_dtype(lhs, rhs)

def check_gemm_onnx(call):
    return True

def check_batch_matmul(call):
    """Check if the given batch_matmul workload can be offloaded to CUTLASS."""
    batch_matmul = get_root_call(call, "nn.batch_matmul")
    lhs = batch_matmul.args[0].checked_type
    rhs = batch_matmul.args[1].checked_type
    transpose_a = batch_matmul.attrs.transpose_a
    transpose_b = batch_matmul.attrs.transpose_b
    return check_dtype(lhs, rhs) and ((not transpose_a and transpose_b) or (not transpose_a and not transpose_b))

def check_batch_matmul_pytorch(call):
    matmul = get_root_call(call, "nn.batch_matmul")
    transpose_dim = get_transpose(matmul.args[1], "transpose")
    
    axes_index = len(transpose_dim) - 1
    
    transpose_a = matmul.attrs.transpose_a
    transpose_b = matmul.attrs.transpose_b
    
    if len(transpose_dim) < 3:
        return (not transpose_a) and (not transpose_b)
    
    
    return False

def check_batch_matmul_pytorch2(call):
    matmul = get_root_call(call, "nn.batch_matmul")
    transpose_a = matmul.attrs.transpose_a
    transpose_b = matmul.attrs.transpose_b
    
    return (not transpose_a) and (not transpose_b)


def is_depthwise_conv2d(ic, oc, groups):
    return ic == oc == groups


def check_conv2d_common(op_name, expected_kernel_layout, call):
    """Check if the given conv2d workload can be offloaded to CUTLASS."""
    conv2d = get_root_call(call, op_name)
    data_layout = conv2d.attrs.data_layout
    kernel_layout = conv2d.attrs.kernel_layout
    data = conv2d.args[0].checked_type
    weight = conv2d.args[1].checked_type
    if (
        data_layout != "NHWC"
        or kernel_layout != expected_kernel_layout
        or not check_dtype(data, weight)
    ):
        return False
    IC = data.shape[3]
    OC = weight.shape[0]
    return not is_depthwise_conv2d(IC, OC, conv2d.attrs.groups)


def check_conv2d(call):
    return check_conv2d_common("nn.conv2d", "OHWI", call)


def check_conv2d_transpose(call):
    # conv2d_transpose is implemented as dgrad, needs to swap the roles of C and K
    return check_conv2d_common("nn.conv2d_transpose", "IHWO", call)


def check_conv2d_backward_weight(call):
    return check_conv2d_common("nn.conv2d_backward_weight", "NHWC", call)


def check_conv2d_residual(call, binary_op):
    """Check if the given conv2d workload can be offloaded to CUTLASS."""
    conv2d = get_root_call(call, "nn.conv2d")
    if not check_conv2d(call):
        return False

    residual_binop = get_root_call(call, binary_op)
    lhs = residual_binop.args[0]
    rhs = residual_binop.args[1]

    # residual_input is pattern-matched as a wildcard. Make sure it does not sit between
    # residual binary op and the root conv2d of this pattern.
    # If the root conv2d is the parent of both lhs and rhs, we should reject this pattern.
    if get_root_call(lhs, "nn.conv2d") == conv2d and get_root_call(rhs, "nn.conv2d") == conv2d:
        return False

    return all(x == y for (x, y) in zip(lhs.checked_type.shape, rhs.checked_type.shape))

def cutlass_test():
    data = wildcard()
    
    return is_op("nn.softmax")(data)

def cutlass_test_check(call):
    return True

def fmha_meta_llama_pattern(dtype="float16"):
    query = wildcard()
    key = wildcard()
    value = wildcard()
    mask = wildcard()
    
    scale_1 = wildcard()
    scale_2 = wildcard()
    
def fmha_llama_pattern_fp32(dtype="float32"):
    query = wildcard()
    key = wildcard()
    value = wildcard()
    mask = wildcard()
    
    scale_1 = wildcard()
    scale_2 = wildcard()
    
    query = is_op("multiply")(query, scale_1)
    query = is_op("reshape")(query)
    
    key = is_op("transpose")(key)
    key = is_op("multiply")(key, scale_2)
    key = is_op("reshape")(key)
    key = is_op("transpose")(key)
    
    qk = is_op("nn.batch_matmul")(query, key)
    qk = is_op("reshape")(qk)
    qk = is_op("add")(qk, mask)
    
    qk = is_op("nn.softmax")(qk)
    qk = is_op("reshape")(qk)
    
    
    value = is_op("reshape")(value)
    value = is_op("transpose")(value)
    qkv = is_op("nn.batch_matmul")(qk, value)
    qkv = is_op("reshape")(qkv)
    qkv = is_op("transpose")(qkv)
    
    return qkv
        

def fmha_llama_pattern(dtype="float16"):
    query = wildcard()
    key = wildcard()
    value = wildcard()
    mask = wildcard()
    
    scale_1 = wildcard()
    scale_2 = wildcard()
    
    scale_1 = is_op("cast")(scale_1)
    query = is_op("multiply")(query, scale_1)
    query = is_op("reshape")(query)
    
    scale_2 = is_op("cast")(scale_2)
    key = is_op("transpose")(key)
    key = is_op("multiply")(key, scale_2)
    key = is_op("reshape")(key)
    key = is_op("transpose")(key)
    
    qk = is_op("nn.batch_matmul")(query, key)
    qk = is_op("reshape")(qk)
    
    mask = is_op("cast")(mask)
    qk = is_op("add")(qk, mask)
    qk = is_op("cast")(qk)
    qk = is_op("nn.softmax")(qk)
    qk = is_op("reshape")(qk)
    qk = is_op("cast")(qk)
    
    
    value = is_op("reshape")(value)
    value = is_op("transpose")(value)
    qkv = is_op("nn.batch_matmul")(qk, value)
    qkv = is_op("reshape")(qkv)
    qkv = is_op("transpose")(qkv)
    
    return qkv
    
def fmha_pattern(dtype="float32"):
    query = wildcard()
    key = wildcard()
    mask = wildcard()
    
    scale_1 = wildcard()
    scale_2 = wildcard()
    
    
    q_proj = is_op("transpose")(query)
    q_proj = is_op("multiply")(q_proj, scale_1)
    q_proj = is_op("reshape")(q_proj)
    
    k_proj = is_op("transpose")(key)
    k_proj = is_op("multiply")(k_proj, scale_2)
    k_proj = is_op("reshape")(k_proj)
    k_proj = is_op("transpose")(k_proj)
    
    qk = is_op("nn.batch_matmul")(q_proj, k_proj)
    qk = is_op("reshape")(qk)
    qk = is_op("add")(qk, mask)
    
    if dtype == "float16":
        qk = is_op("cast")(qk)
    
    qk = is_op("nn.softmax")(qk)
    qk = is_op("reshape")(qk)
    
    if dtype == "float16":
        qk = is_op("cast")(qk)
    
    value = wildcard()
    
    # v_proj = is_op("reshape")(input_tensor)
    # v_proj = is_op("transpose")(value).has_attr({"axes": [0, 2, 1, 3]})
    v_proj = is_op("reshape")(value)
    v_proj = is_op("transpose")(v_proj)      
    
    
    qkv = is_op("nn.batch_matmul")(qk, v_proj)
    qkv = is_op("reshape")(qkv)
    qkv = is_op("transpose")(qkv)
    
    return qkv

def fmha_bert_pattern(dtype="float32"):
    query = wildcard()
    key = wildcard()
    mask = wildcard()
    
    scale_1 = wildcard()
    scale_2 = wildcard()
    
    
    q_proj = is_op("transpose")(query)
    q_proj = is_op("multiply")(q_proj, scale_1)
    q_proj = is_op("reshape")(q_proj)
    
    k_proj = is_op("transpose")(key)
    k_proj = is_op("multiply")(k_proj, scale_2)
    k_proj = is_op("reshape")(k_proj)
    k_proj = is_op("transpose")(k_proj)
    
    qk = is_op("nn.batch_matmul")(q_proj, k_proj)
    qk = is_op("reshape")(qk)

    if dtype == "float16":
        qk = is_op("cast")(qk)
    qk = is_op("add")(qk, mask)
    
    qk = is_op("nn.softmax")(qk)
    qk = is_op("reshape")(qk)
    
    if dtype == "float16":
        qk = is_op("cast")(qk)
        
    value = wildcard()
    
    v_proj = is_op("reshape")(value)
    v_proj = is_op("transpose")(v_proj)      
    
    
    qkv = is_op("nn.batch_matmul")(qk, v_proj)
    qkv = is_op("reshape")(qkv)
    qkv = is_op("transpose")(qkv)
    
    return qkv

def fmha_pattern2(dtype="float32"):
    query = wildcard()
    key = wildcard()
    mask = wildcard()
    
    q_proj = is_op("transpose")(query)
    q_proj = is_op("reshape")(q_proj)
    
    k_proj = is_op("transpose")(key)
    k_proj = is_op("reshape")(k_proj)
    k_proj = is_op("transpose")(k_proj)
    
    qk = is_op("nn.batch_matmul")(q_proj, k_proj)
    qk = is_op("reshape")(qk)
    qk = is_op("divide")(qk, is_constant() | wildcard())
    
    if dtype == "float16":
        qk = is_op("cast")(qk)
    
    qk = is_op("add")(qk, mask)
    
    qk = is_op("nn.softmax")(qk)
    qk = is_op("reshape")(qk)
    
    if dtype == "float16":
        qk = is_op("cast")(qk)
    
    value = wildcard()
    
    # v_proj = is_op("reshape")(input_tensor)
    # v_proj = is_op("transpose")(value).has_attr({"axes": [0, 2, 1, 3]})
    v_proj = is_op("reshape")(value)
    v_proj = is_op("transpose")(v_proj)      
    
    
    qkv = is_op("nn.batch_matmul")(qk, v_proj)
    qkv = is_op("reshape")(qkv)
    qkv = is_op("transpose")(qkv)
    
    return qkv

def fmha_gpt_pattern2(dtype="float32"):
    query = wildcard()
    key = wildcard()
    mask = wildcard()
    
    q_proj = is_op("transpose")(query)
    q_proj = is_op("reshape")(q_proj)
    
    k_proj = is_op("transpose")(key)
    k_proj = is_op("reshape")(k_proj)
    k_proj = is_op("transpose")(k_proj)
    
    qk = is_op("nn.batch_matmul")(q_proj, k_proj)
    qk = is_op("reshape")(qk)
    qk = is_op("divide")(qk, is_constant() | wildcard())
    qk = is_op("cast")(qk)
    qk = is_op("where")(is_constant() | wildcard(), qk, is_constant() | wildcard())
    
    qk = is_op("nn.softmax")(qk)
    qk = is_op("cast")(qk)
    qk = is_op("reshape")(qk)
    
    if dtype == "float16":
        qk = is_op("cast")(qk)
    
    value = wildcard()
    
    # v_proj = is_op("reshape")(input_tensor)
    # v_proj = is_op("transpose")(value).has_attr({"axes": [0, 2, 1, 3]})
    v_proj = is_op("reshape")(value)
    v_proj = is_op("transpose")(v_proj)      
    
    
    qkv = is_op("nn.batch_matmul")(qk, v_proj)
    qkv = is_op("reshape")(qkv)
    qkv = is_op("transpose")(qkv)
    
    return qkv

def tmp():
    return wildcard()(wildcard(), wildcard())

def check_fmha(call):
    
    return True
    
@register_pattern_table("curator")
def pattern_table():
    """Returns list of triples describing the name, dataflow pattern and predicate for all
    the CUTLASS-supported operators."""
    
    #FMHA
    fmha_fp32_pat = ("curator.fmha_fp32", fmha_pattern(), check_fmha)
    fmha_fp16_pat = ("curator.fmha_fp16", fmha_pattern(dtype="float16"), check_fmha)
    fmha_llama_fp16_pat = ("curator.fmha_openllama_fp16", fmha_llama_pattern(), check_fmha)
    fmha_llama_fp32_pat = ("curator.fmha_openllama_fp32", fmha_llama_pattern_fp32(), check_fmha)
    fmha_bert_fp16_pat = ("curator.fmha_bert_fp16", fmha_bert_pattern(dtype="float16"), check_fmha)
    fmha_bert_fp32_pat = ("curator.fmha_bert_fp32", fmha_bert_pattern(dtype="float32"), check_fmha)
    
    # Pytorch v1.12.1
    fmha_bert_fp16_path2 = ("curator.fmha2_bert_fp16", fmha_pattern2(dtype="float16"), check_fmha)
    fmha_bert_fp32_path2 = ("curator.fmha2_bert_fp32", fmha_pattern2(), check_fmha)
    
    fmha_gpt2_fp16_path2 = ("curator.fmha2_gpt_fp16", fmha_gpt_pattern2(dtype="float16"), check_fmha)
    fmha_gpt2_fp32_path2 = ("curator.fmha2_gpt_fp32", fmha_gpt_pattern2(), check_fmha)
    
    #GEMM
    dense_pat = ("curator.dense", make_gemm_pattern(False, None), check_gemm)
    dense_bias_pat = ("curator.dense_bias", make_gemm_pattern(True, None), check_gemm)
    dense_bias_relu_pat = ("curator.dense_bias_relu", make_gemm_pattern(True, "relu"), check_gemm)
    dense_bias_gelu_fp16_pat = (
        "curator.dense_bias_gelu_fp16",
        make_gemm_pattern(True, "gelu"),
        check_gemm,
    )
    dense_bias_gelu_fp32_pat = (
        "curator.dense_bias_gelu_fp32",
        make_gemm_pattern(True, "gelu", out_dtype="float32"),
        check_gemm,
    )
    dense_bias_pat_onnx = ("curator.onnx_dense_bias", make_gemm_pattern_onnx(True, None), check_gemm_onnx)
    dense_bias_relu_pat_onnx = ("curator.onnx_dense_bias_relu", make_gemm_pattern_onnx(True, "relu"), check_gemm_onnx)
    dense_bias_gelu_fp32_pat_onnx = ("curator.onnx_dense_bias_gelu_fp32", make_gemm_pattern_onnx(True, "gelu", out_dtype="float32"), check_gemm_onnx)
    dense_bias_gelu_fp16_pat_onnx = ("curator.onnx_dense_bias_gelu_fp16", make_gemm_pattern_onnx(True, "gelu", out_dtype="float16"), check_gemm_onnx)
    
    
    #batch GEMM
    batch_matmul_pat = ("curator.batch_matmul", make_batch_matmul_pattern_custom(False, None), check_batch_matmul)
    batch_matmul_bias_pat = ("curator.batch_matmul_bias", make_batch_matmul_pattern_custom(True, None), check_batch_matmul)
    batch_matmul_bias_relu_pat = ("curator.batch_matmul_bias_relu", make_batch_matmul_pattern_custom(True, "relu"), check_batch_matmul)
    
    batch_matmul_gelu_fp32_pat = ("curator.batch_matmul_bias_gelu_fp32", make_batch_matmul_pattern_custom(True, "gelu", out_dtype="float32"), check_batch_matmul)
    batch_matmul_gelu_fp16_pat = ("curator.batch_matmul_bias_gelu_fp16", make_batch_matmul_pattern_custom(True, "gelu", out_dtype="float16"), check_batch_matmul)
    
    
    # batch_matmul_pytorch_transpose_pat = ("cutlass.batch_matmul_pytorch_transpose", make_batch_matmul_pattern_pytorch(False, None, True), check_batch_matmul_pytorch)
    # batch_matmul_bias_pytorch_transpose_pat = ("cutlass.batch_matmul_bias_pytorch_transpose", make_batch_matmul_pattern_pytorch(True, None, True), check_batch_matmul_pytorch)
    # batch_matmul_bias_relu_pytorch_transpose_pat = ("cutlass.batch_matmul_bias_relu_pytorch_transpose", make_batch_matmul_pattern_pytorch(True, "relu", True), check_batch_matmul_pytorch)
    # batch_matmul_gelu_fp32_pytorch_transpose_pat = ("cutlass.batch_matmul_bias_gelu_fp32_pytorch_transpose", make_batch_matmul_pattern_pytorch(True, "gelu", True), check_batch_matmul_pytorch)
    
    # batch_matmul_pytorch_pat = ("cutlass.batch_matmul_pytorch", make_batch_matmul_pattern_pytorch(False, None, False), check_batch_matmul_pytorch2)
    # batch_matmul_bias_pytorch_pat = ("cutlass.batch_matmul_bias_pytorch", make_batch_matmul_pattern_pytorch(True, None, False), check_batch_matmul_pytorch2)
    # batch_matmul_bias_relu_pytorch_pat = ("cutlass.batch_matmul_bias_relu_pytorch", make_batch_matmul_pattern_pytorch(True, "relu", False), check_batch_matmul_pytorch2)
    # batch_matmul_gelu_fp32_pytorch_pat = ("cutlass.batch_matmul_bias_gelu_fp32_pytorch", make_batch_matmul_pattern_pytorch(True, "gelu", False), check_batch_matmul_pytorch2)
    
    #softmax
    softmax_support_pat = ("curator.softmax_support", cutlass_test(), cutlass_test_check)
    
    dense_patterns = [
        #Fused multi head attention
        fmha_fp16_pat,
        fmha_fp32_pat,
        fmha_llama_fp16_pat,
        fmha_llama_fp32_pat,
        fmha_bert_fp16_pat,
        fmha_bert_fp32_pat,
        
        fmha_bert_fp32_path2,
        fmha_bert_fp16_path2,
        
        fmha_gpt2_fp16_path2,
        fmha_gpt2_fp32_path2,
        
        # just dense
        dense_bias_gelu_fp32_pat_onnx,
        dense_bias_gelu_fp16_pat,
        dense_bias_gelu_fp16_pat_onnx,
        dense_bias_gelu_fp32_pat,
        dense_bias_relu_pat_onnx,
        dense_bias_relu_pat,
        dense_bias_pat_onnx,
        dense_bias_pat,
        dense_pat,
        # ("cutlass.batch_matmul", make_batch_matmul_pattern(), check_batch_matmul),
        
        #just batch matmul
        batch_matmul_gelu_fp16_pat,
        batch_matmul_gelu_fp32_pat,
        batch_matmul_bias_relu_pat,
        batch_matmul_bias_pat,
        batch_matmul_pat,
        
        #support cutlass
        softmax_support_pat,
    ]


    return dense_patterns


def partition_for_curator(mod, params=None, fmha=False):
    """Partition the input module into CUTLASS-supported subgraphs."""
    
    if params is not None:
        mod["main"] = bind_params_by_name(mod["main"], params)
        remove_bn_pass = Sequential(
            [
                transform.InferType(),
                transform.SimplifyInference(),
                transform.FoldConstant(),
                transform.FoldScaleAxis(),
            ]
        )
        with PassContext(opt_level=3):
            mod = remove_bn_pass(mod)

    cutlass_patterns = relay.op.contrib.get_pattern_table("curator")
    
    new_cutlass_patterns = []
    
    for pattern in cutlass_patterns:
        if fmha == False and "fmha" in str(pattern[0]):
            continue
        new_cutlass_patterns.append(pattern)
    
    seq = Sequential(
        [
            transform.InferType(),
            transform.MergeComposite(new_cutlass_patterns),
            transform.AnnotateTarget(["curator"], include_non_call_ops=False),
            transform.PartitionGraph(bind_constants=False),
        ]
    )

    return seq(mod)
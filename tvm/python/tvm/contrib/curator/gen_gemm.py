from .profile_cutlass_gemm import ProfileBatchedGemm, ProfileGemm
from .profile_cutlass_gemm_tc import ProfileBatchedGemmTC, ProfileGemmTC
from .gemm_operation import GemmOperation, EmitGemmInstance
from .gen_tensor_op import EPILOGUE_MAP

from .library import (
    DataType,
    DataTypeTag,
    EpilogueFunctor,
    LayoutType,
    SwizzlingFunctor,
    TensorDescription,
    TileDescription,
    MathInstruction,
    DataType,
    OpcodeClass,
    MathOperation,
)

import json
import os

def enumerate_gemm_operators(
    tile_descriptions,
    data_type,
    alignment_constraints,
    C_alignment_constraints=1,
    swizzling_functor=SwizzlingFunctor.Identity8,
    transpose_a=False,
    transpose_b=True,
):
    """Exhaustively instantiate all kernels from a given configuration."""
    ret = []

    element_a, element_b, element_c, element_epilogue = data_type
    
    for tile_description in tile_descriptions:
        for alignment in alignment_constraints:
            if transpose_a == True:
                A = TensorDescription(element_a, LayoutType.ColumnMajor, alignment)
            else:
                A = TensorDescription(element_a, LayoutType.RowMajor, alignment)
            
            if transpose_b == True:
                B = TensorDescription(element_b, LayoutType.ColumnMajor, alignment)
            else:
                B = TensorDescription(element_b, LayoutType.RowMajor, alignment)
            
            C = TensorDescription(element_c, LayoutType.RowMajor, C_alignment_constraints)
            
            if element_c == DataType.s32 and A.alignment == 1:
                tile_description.threadblock_shape[0] = min(
                    tile_description.threadblock_shape[0], 128
                )
                tile_description.threadblock_shape[1] = min(
                    tile_description.threadblock_shape[1], 128
                )
            
            op = GemmOperation(
                tile_description.minimum_compute_capability,
                tile_description,
                A,
                B,
                C,
                element_epilogue,
                EpilogueFunctor.LinearCombination,
                swizzling_functor,
            )
            
            ret.append(
                {
                    "op": op,
                    "name": op.procedural_name(),
                    "tile_description": tile_description,
                    "alignment": alignment,
                    "c_alignment": C_alignment_constraints,
                    "data_type": data_type,
                    "swizzle_functor": swizzling_functor,
                    "split_k": tile_description.split_k
                }
            )
    return ret

def create_gemm_operator_with_epilogue(
    op_type,
    tile_description,
    data_type,
    alignment,
    c_alignment,
    swizzling_functor,
    split_k=1,
    batched=False,
    transpose_a=False,
    transpose_b=True,
):
    element_a, element_b, element_c, element_epilogue = data_type
    
    if transpose_a == True:
        A = TensorDescription(element_a, LayoutType.ColumnMajor, alignment)
    else:
        A = TensorDescription(element_a, LayoutType.RowMajor, alignment)
    
    if transpose_b == True:
        B = TensorDescription(element_b, LayoutType.ColumnMajor, alignment)
    else:
        B = TensorDescription(element_b, LayoutType.RowMajor, alignment)
    
    C = TensorDescription(element_c, LayoutType.RowMajor, c_alignment)
    
    if batched:
        swizzling_functor = SwizzlingFunctor.Batched
    
    epilogue, no_beta_scaling = EPILOGUE_MAP[op_type]
    
    op = GemmOperation(
        tile_description.minimum_compute_capability,
        tile_description,
        A,
        B,
        C,
        element_epilogue,
        epilogue,
        swizzling_functor,
    )
    return (
        op.procedural_name(),
        EmitGemmInstance().emit(op, no_beta_scaling=no_beta_scaling, split_k=split_k, batched=batched),
    )

class CutlassGemmProfiler:
    def __init__(self, tuning_config):
        self.tuning_config = tuning_config
        self.cache = {}
    
    def select_op(
        self,
        M,
        N,
        K,
        out_dtype,
        arg0_dtype,
        arg1_dtype,
        batch=False,
        batch_count=1,
        transpose_a=False,
        transpose_b=True,
        op_type=""
    ):
        if (batch_count, M, N, K) in self.cache:
            op = self.cache[(batch_count, M, N, K)]
            return op
        
        C_alignment_constraints = 1
        if arg0_dtype == "float32" and arg1_dtype == "float32" and out_dtype == "float32":
            math_instruction = [
                    MathInstruction(
                        [1, 1, 1],
                        DataType.f32,
                        DataType.f32,
                        DataType.f32,
                        DataType.f32,
                        OpcodeClass.Simt,
                        MathOperation.multiply_add,
                    ),
                ]
            alignment_constraints = [1,]
            
            if batch == True:
                gemm_profile = ProfileBatchedGemm(batch=batch_count, M=M, N=N, K=K, tuning_config=self.tuning_config)
                tile, split, sizzle = gemm_profile.eval_cutlassOracle(transpose_a=transpose_a, transpose_b=transpose_b)
            else:
                gemm_profile = ProfileGemm(batch=batch_count, M=M, N=N, K=K, tuning_config=self.tuning_config)
                tile, split, sizzle = gemm_profile.eval_cutlassOracle(transpose_a=transpose_a, transpose_b=transpose_b)
            
            sm = int(self.tuning_config["sm"])
            sm = 86 if sm == 89 else sm
            
        elif arg0_dtype == "float16" and arg1_dtype == "float16" and out_dtype == "float16":
            math_instruction = [
                    MathInstruction(
                        [16, 8, 16],
                        DataType.f16,
                        DataType.f16,
                        DataType.f16,
                        DataType.f16,
                        OpcodeClass.TensorOp,
                        MathOperation.multiply_add,
                    ),
                ]
            
            if batch == True:
                gemm_profile = ProfileBatchedGemmTC(batch=batch_count, M=M, N=N, K=K, tuning_config=self.tuning_config)
                tile, split, sizzle = gemm_profile.eval_cutlassOracle(transpose_a=transpose_a, transpose_b=transpose_b)
            else:
                gemm_profile = ProfileGemmTC(batch=batch_count, M=M, N=N, K=K, tuning_config=self.tuning_config)
                tile, split, sizzle = gemm_profile.eval_cutlassOracle(transpose_a=transpose_a, transpose_b=transpose_b)
            
            C_alignment_constraints = 8 if int(N) % 8 == 0 else 4
            
            alignment_constraints = tile[3]
            sm = int(self.tuning_config["sm"])
            sm = 80 if sm == 89 or sm == 86 else sm
        else:
            assert 0 == 10, "Error you shoud compute float32 * float32 + float32 or float16 * float16 + float16"
            
        # tile description
        block_tile = tile[0]
        buffer_stage = tile[2][0]
        warp_tile = [int(tile[0][0] / tile[1][0]), int(tile[0][1] / tile[1][1]), int(tile[0][2] / tile[1][2])]        
        tile_descriptions  = [(block_tile, buffer_stage, warp_tile, split, sm, 1024),]
        
        description_all = [
                TileDescription(threadblock_shape, stages, warp_count, math_instruction[0], min_cc, max_cc, split_k=split_k)
                for threadblock_shape, stages, warp_count, split_k, min_cc, max_cc in tile_descriptions
            ]
            
        data_dtype = [
            math_instruction[0].element_a,
            math_instruction[0].element_b,
            math_instruction[0].element_c,
            math_instruction[0].element_accumulator,
        ]   
        
        if sizzle == 8:
                sizzle_func = SwizzlingFunctor.Identity8
        elif sizzle == 4:
            sizzle_func = SwizzlingFunctor.Identity4
        elif sizzle == 2:
            sizzle_func = SwizzlingFunctor.Identity2
        elif sizzle == 1:
            sizzle_func = SwizzlingFunctor.Identity1
        
        out_ops = enumerate_gemm_operators(description_all, data_dtype, alignment_constraints, C_alignment_constraints=C_alignment_constraints, transpose_a=transpose_a, transpose_b=transpose_b, swizzling_functor=sizzle_func)
        out_ops[0]["runtime"] = 0.0
        self.cache[(batch_count, M, N, K)] = out_ops[0]
        
        return out_ops[0]
    
    def profile(
        self,
        op_type,
        M,
        N,
        K,
        out_dtype,
        arg0_dtype,
        arg1_dtype,
        batched=False,
        batch_count=1,
        transpose_a=False,
        transpose_b=True,
    ):
        op = self.select_op(
            M,
            N,
            K,
            out_dtype,
            arg0_dtype,
            arg1_dtype,
            batched,
            batch_count,
            transpose_a,
            transpose_b,
            op_type)
        
        name, opdef = create_gemm_operator_with_epilogue(
            op_type,
            op["tile_description"],
            op["data_type"],
            op["alignment"],
            op["c_alignment"],
            op["swizzle_functor"],
            split_k=op["split_k"],
            batched=batched,
            transpose_a=transpose_a,
            transpose_b=transpose_b
        )
        
        return name, opdef, op["split_k"]
    
    
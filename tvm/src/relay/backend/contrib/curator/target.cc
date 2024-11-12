/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relay/backend/contrib/cutlass/target.cc
 * \brief Registers the "cutlass" external codegen TargetKind.
 */

#include <tvm/target/target.h>

#include "./codegen.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace curator {

/*!
 * \brief This external codegen target can use the CUTLASS template library included in
 * TVM's 3rdparty/cutlass.
 *  - Patterns: python/tvm/relay/op/contrib/cutlass.py
 *  - Custom compiler: python/tvm/contrib/cutlass/build.py,
 *                     src/relay/backend/contrib/cutlass/codegen.cc
 */
TVM_REGISTER_TARGET_KIND("curator", kDLCUDA)
    .set_attr<Bool>(tvm::attr::kIsExternalCodegen, Bool(true))
    .set_attr<FTVMRelayToTIR>("RelayToTIR", CompileForCutlass())
    // An integer specifying the compute capability. For example, 75 for Turing and
    // 80 or 86 for Ampere.
    .add_attr_option<Integer>("sm", Integer(80))
    //tbt m range
    .add_attr_option<Array<Integer>>("tbt_m", Array<Integer>({1}))
    //tbt n range
    .add_attr_option<Array<Integer>>("tbt_n", Array<Integer>({1}))
    //tbt k range
    .add_attr_option<Array<Integer>>("tbt_k", Array<Integer>({1}))
    // software pipelining stage
    .add_attr_option<Array<Integer>>("pipelining_range", Array<Integer>({1}))
    //alignment stage
    .add_attr_option<Array<Integer>>("align_range", Array<Integer>({1}))
    // range of split-k
    .add_attr_option<Array<Integer>>("split_k_range", Array<Integer>({1}))
    // range of swizzle
    .add_attr_option<Array<Integer>>("swizzle_range", Array<Integer>({1}))
    // Whether to replace sigmoid with tanh.
    .add_attr_option<Bool>("use_fast_math", Bool(false))
    // A temporary directory where intermediate compiled artifacts will be stored.
    .add_attr_option<String>("tmp_dir", String("./tmp"));

}  // namespace cutlass
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

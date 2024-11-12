#include "flash.h"
#include "static_switch.h"

extern "C"{
// void run_mha_fwd64_64(Flash_fwd_params &params, bool force_split_kernel){
//   FP16_SWITCH(!params.is_bf16, [&] {
//     HEADDIM_SWITCH(params.d, [&] {
//       BOOL_SWITCH(params.is_causal, Is_causal, [&] {
//         if(params.num_splits <= 1 && !force_split_kernel){
//         //   run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params);
//           run_mha_fwd_hdim64_64<elem_type, kHeadDim, Is_causal>(params);
//         } else{
//           // run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params);
//         }
//       });
//     });
//   });
// }

// void run_mha_fwd128_128(Flash_fwd_params &params, bool force_split_kernel){
//   FP16_SWITCH(!params.is_bf16, [&] {
//     HEADDIM_SWITCH(params.d, [&] {
//       BOOL_SWITCH(params.is_causal, Is_causal, [&] {
//         if(params.num_splits <= 1 && !force_split_kernel){
//         //   run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params);
//           run_mha_fwd_hdim128_128<elem_type, kHeadDim, Is_causal>(params);
//         } else{
//           // run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params);
//         }
//       });
//     });
//   });
// }

void run_mha_fwd_splitk(Flash_fwd_params &params, bool force_split_kernel){
  FP16_SWITCH(!params.is_bf16, [&] {
    HEADDIM_SWITCH(params.d, [&] {
      BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        if(params.num_splits <= 1 && !force_split_kernel){
        //   run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params);
        //   run_mha_fwd_hdim64_64<elem_type, kHeadDim, Is_causal>(params);
        } else{
          run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params);
        }
      });
    });
  });
}



}
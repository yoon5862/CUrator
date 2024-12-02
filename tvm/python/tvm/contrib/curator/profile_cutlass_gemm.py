import os
import multiprocessing

import re
import json

class BatchedGEMMTemplate:
    def __init__(self):
        self.template = """
#include<iostream>
#include<cuda_runtime.h>

#include <unistd.h>
#include<string>
#include<fstream>      

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm_batched_splitK.h"
#include <cutlass/util/host_tensor.h>

${function_body}

int main(int argc, char *argv[]){
    float *rlt = new float[${opt_rlt}];

    int M = 64;
    int N = 64;
    int K = 64;
    int Batch = 1;
    int split_k = 1;
    int device_id;
    
    int option;
    while((option = getopt(argc, argv, "m:n:k:b:s:d:")) != -1){
        switch(option){
            case 'm':
                M = std::stoi(optarg);
                break;
            case 'n':
                N = std::stoi(optarg);
                break;
            case 'k':
                K = std::stoi(optarg);
                break;
            case 'b':
                Batch = std::stoi(optarg);
                break;
            case 's':
                split_k = std::stoi(optarg);
            case 'd':
                device_id = std::stoi(optarg);
                break;
            case '?':
                break;
        }
    }
    cudaSetDevice(device_id);
    
    int const lda = ${lda};
    int const ldb = ${ldb};
    int const ldc = ${ldc};
    
    int const count_A = Batch * M * K;
    int const count_B = Batch * N * K;
    int const count_C = Batch * M * N;
    
    long long int batch_stride_A = static_cast<long long int>(M) * static_cast<long long int>(K);
    long long int batch_stride_B = static_cast<long long int>(K) * static_cast<long long int>(N);
    long long int batch_stride_C = static_cast<long long int>(M) * static_cast<long long int>(N);
    
    ${outType_1} alpha = static_cast<${outType_1}>(1.0f);
    ${outType_2} beta = static_cast<${outType_2}>(0.0f);
    
    std::vector<${outType_1}> host_A(count_A, 1.2f);
    std::vector<${outType_1}> host_B(count_B, 1.0f);
    std::vector<${outType_2}> host_C(count_C);
    
    ${outType_1} *A;
    ${outType_1} *B;
    ${outType_2} *C;
    
    cudaMalloc(&A, count_A * sizeof(${outType_1}));
    cudaMalloc(&B, count_B * sizeof(${outType_1}));
    cudaMalloc(&C, count_C * sizeof(${outType_2}));
    
    cudaMemcpy(A, host_A.data(), count_A * sizeof(${outType_1}), cudaMemcpyHostToDevice);
    cudaMemcpy(B, host_B.data(), count_B * sizeof(${outType_1}), cudaMemcpyHostToDevice);
    cudaMemcpy(C, host_C.data(), count_C * sizeof(${outType_2}), cudaMemcpyHostToDevice);
    
    //warmp up
    for(int i = 0; i < 2; i++){
        cutlass_strided_bathed_sgemm_0(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, 1, 1);
    }
    
    ${exec_body}
    
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    
    return 0;
}
        """
        
        self.function_template = """
float cutlass_strided_bathed_sgemm_${number}(
    int m, int n, int k,
    ${outType_1} alpha, ${outType_1} const *A, int lda, long long int batch_stride_A,
    ${outType_1} const *B, int ldb, long long int batch_stride_B,
    ${outType_2} *C, int ldc, long long int batch_stride_C,
    ${outType_2} beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<${outType_1}, ${layout_0},
                                                    ${outType_1}, ${layout_1},
                                                    ${outType_2}, ${layout_2},
                                                    ${outType_2},
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm${sm},
                                                    cutlass::gemm::GemmShape<${MM}, ${MN}, ${MK}>,
                                                    cutlass::gemm::GemmShape<${WM}, ${WN}, ${WK}>,
                                                    cutlass::gemm::GemmShape<1, 1, 1>,
                                                    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
                                                    cutlass::gemm::threadblock::GemmBatchedSplitKIdentityThreadblockSwizzle,
                                                    ${Stage},
                                                    ${Al},
                                                    ${Al},
                                                    true,
                                                    cutlass::arch::OpMultiplyAdd
                                                    >;
    
    Gemm gemm_op;
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    
    Gemm::Arguments arguments{
            {m, n, k},
            {A, lda}, batch_stride_A,
            {B, ldb}, batch_stride_B,
            {C, ldc}, batch_stride_C,
            {C, ldc}, batch_stride_C,
            {alpha, beta},
            batch_count,
            ${split_k}
        };
        
    int *workspace;
    int workspace_size = ((m + (${MM} - 1)) / ${MM}) * ((m + (${MN} - 1)) / ${MN}) * batch_count * sizeof(int);
    cudaMalloc(&workspace, workspace_size);
    cudaMemset(workspace, 0, workspace_size);
    
    cutlass::Status status = gemm_op.can_implement(arguments);
    if(status != cutlass::Status::kSuccess) return -1;
    
    status = gemm_op.initialize(arguments, workspace);
    if(status != cutlass::Status::kSuccess) return -1;
    
    cudaEventRecord(start);
    
    for(int i = 0; i < ${repeat}; i++){
        status = gemm_op();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "${rlt_json_dir}/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\\"dim\\": [[${MM}, ${MN}, ${MK}], [${WM}, ${WN}, ${WK}], [${Stage}], [${Al}]], \\"split_k\\": " + std::to_string(split_k)
                                    + " ,\\"time\\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }
            cudaFree(workspace);
            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);
    
    cudaFree(workspace);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "${rlt_json_dir}/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\\"dim\\": [[${MM}, ${MN}, ${MK}], [${WM}, ${WN}, ${WK}], [${Stage}], [${Al}]], \\"split_k\\": " + std::to_string(split_k)
                            + " ,\\"time\\": " + std::to_string(total_time/${repeat}) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / ${repeat};
    
}
        """
    
    def cutlass_gemm_func(self, opt_rlt=[], number=0, repeat=20, transpose_a=False, transpose_b=True, rlt_json_dir="", opt_split_k="split_k", sm=70):
        shared_mem = opt_rlt[0]
        reg_mem = opt_rlt[1]
        stages = opt_rlt[2][0]
        alignment = opt_rlt[3][0]
        
        template_value = {}
        template_value["MM"] = str(shared_mem[0])
        template_value["MN"] = str(shared_mem[1])
        template_value["MK"] = str(shared_mem[2])
        
        template_value["WM"] = str(reg_mem[0])
        template_value["WN"] = str(reg_mem[1])
        template_value["WK"] = str(reg_mem[2])
        
        template_value["Stage"] = str(stages)
        template_value["Al"] = str(alignment)
        template_value["repeat"] = str(repeat)
        template_value["number"] = str(number)
        
        template_value["layout_0"] = "cutlass::layout::ColumnMajor" if transpose_a == True else "cutlass::layout::RowMajor"
        template_value["layout_1"] = "cutlass::layout::ColumnMajor" if transpose_b == True else "cutlass::layout::RowMajor"
        template_value["layout_2"] = "cutlass::layout::RowMajor"
        template_value["rlt_json_dir"] = rlt_json_dir
        
        template_value['split_k'] = opt_split_k
        
        if int(sm) == 89:
            template_value["sm"] = str(86)
        else:
            template_value["sm"] = str(sm)
        
        template = self.substitute_template(self.function_template, template_value)
        return template
    
    def main_pre(self, template, transpose_a=False, transpose_b=False, dtype="float", outtype="float", len=0):
        value = {}
        
        
        value["lda"] = "K" if transpose_a == False else "M"
        value["ldb"] = "N" if transpose_b == False else "K"
        value["ldc"] = "N"
        
        value["outType_1"] = dtype
        value["outType_2"] = outtype
        
        value["opt_rlt"] = str(len)
        
        template = self.substitute_template(template, value)
              
        return template
    
    def main_exec(self, template, len=0):
        value = {}
        for i in range(len):
            value_key = f"exec_body_{str(i)}"
            value[value_key] = f"rlt[{str(i)}] = cutlass_strided_bathed_sgemm_{str(i)}(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);"
            
        template = self.substitute_template(template, value)
        return template
    
    def split_k_exec(self, template, split_k=1):
        value = {'split_k': str(split_k)}
        template = self.substitute_template(template, value)
        return template
    
    def substitute_template(self, template, values):
        text = template
        changed = True
        while changed:
            changed = False
            for key, value in values.items():
                regex = "\\$\\{%s\\}" % key
                newtext = re.sub(regex, value, text)
                if newtext != text:
                    changed = True
                text = newtext
        return text
    
class ProfileBatchedGemm:
    def __init__(self, batch=1, M=64, N=64, K=64, tuning_config={}):
        self.batch = int(batch)
        self.M = int(M)
        self.N = int(N)
        self.K = int(K)
        self.tuning_config = tuning_config
        self.split_k = [int(split_k) for split_k in tuning_config["split_k_range"]]
        self.tailor = False
        self.sm = int(tuning_config["sm"])
        # self.target_node = int(tuning_config["target_node"])
        self.target_node = 0
    
    def create_cutlass_opt(self, m_shape=[], n_shape=[], k_shape=[], kstage_shape=[], alignment=1, transpose_a=False, transpose_b=False):
    # #_shape = [start, end, stride]
    # return option
        compile_option = []
        
        def iswarp(shared_m, shared_n, warp_m, warp_n, kstage=2):
            if((shared_m / warp_m) * (shared_n / warp_n)) > 16:
                return False
            
            if (shared_m % warp_m) != 0 or (shared_n % warp_n) != 0:
                return False
            
            tmp_warp = (shared_m // warp_m) * (shared_n // warp_n) * 32
            
            if tmp_warp > 256:
                return False
            
            if transpose_a == True:
                if tmp_warp % shared_m != 0 or (shared_m % tmp_warp != 0 and tmp_warp < shared_m):
                    return False
            
            if transpose_b == False:
                if tmp_warp % shared_n != 0 or (shared_n % tmp_warp and tmp_warp < shared_n) != 0:
                    return False
            
            warpNumThreadsM = 8
            
            if warp_m <= warp_n:
                warpNumThreadsM = 4
                
            if kstage > 2:
                warpNumThreadsM = 4
                warpNumThreadsN = 8
            
            warpNumThreadsN = 32 / warpNumThreadsM
            
            ThreadTileM = warp_m / warpNumThreadsM
            ThreadTileN = warp_n / warpNumThreadsN
            
            if warp_m % warpNumThreadsM != 0 or warp_n % warpNumThreadsN != 0:
                return False
            
            laneM = min(4, ThreadTileM)
            laneN = min(4, ThreadTileN)
            
            paddingM = 8
            paddingN = 8
            
            if transpose_a == True:
                paddingM = 0
            if transpose_b == False:
                paddingN = 0
            
            if paddingM % laneM != 0 or paddingN % laneN != 0:
                return False
            
            if warp_m / warpNumThreadsM < 0 or warp_n / warpNumThreadsN < 0:
                return False
            
            if warp_m % warpNumThreadsM != 0 or warp_n % warpNumThreadsN != 0:
                return False
            
            iterationA = warp_m / warpNumThreadsM
            iterationB = warp_n / warpNumThreadsN
        
            if iterationA % laneM != 0 or iterationB % laneN != 0:
                return False
            
            return True
        
        for stage in range(kstage_shape[0], kstage_shape[1] + kstage_shape[2], kstage_shape[2]):
            for shared_m in range(m_shape[0], m_shape[1] + m_shape[2], m_shape[2]):
                for shared_n in range(n_shape[0], n_shape[1] + n_shape[2], n_shape[2]):
                    for shared_k in range(k_shape[0], k_shape[1] + k_shape[2], k_shape[2]):
                        for warp_m in range(1, shared_m + 1):
                            for warp_n in range(1, shared_n + 1):
                                if iswarp(shared_m, shared_n, warp_m, warp_n, stage) is False:
                                    continue
                                
                                op = [[shared_m, shared_n, shared_k], [warp_m , warp_n, shared_k], [stage], [alignment]]
                                
                                # if transpose_a == False and transpose_a == False:
                                #      #352, 353 option which cutlass doesn't work
                                #     if op == [[224, 64, 8], [16, 64, 8], [2], [1]] or op == [[224, 64, 8], [32, 32, 8], [2], [1]]:
                                #         continue
                                compile_option.append(op)
        
        return compile_option
    
    def create_tailor_opt(self):
        tailor_dict = {}
        
        real_path = os.path.dirname(__file__)
        with open(f"{real_path}/tailor.txt", "r") as f:
            for line in f:
                line = line.split(' ')
                dict_key = f"{line[6]} {line[7]} {line[8]}"
                
                if len(line[9]) < 4:
                    continue
                
                parameter = line[9].split('_')
                parameter_block = parameter[0].split('x')
                parameter_block = [int(parameter_block[0]), int(parameter_block[1]), 8]
                
                parameter_warp = parameter[1].split("x")
                parameter_warp = [int(parameter_warp[0]), int(parameter_warp[1]), 8]
                
                split_k = parameter[2]
                split_k = split_k.replace("\n", "")
                split_k = int(split_k) if int(split_k) > 0 else 1
                
                if not dict_key in tailor_dict:
                    tailor_dict.update({dict_key: []})
                tailor_dict[dict_key].append([[parameter_block, parameter_warp, [2], [1]], split_k])
                # print(f"{dict_key}: {parameter_block}, {parameter_warp}, {split_k}")
        
        opt_rlt = []
        opt_split_k = []
        dict_key = f"{self.M}, {self.batch * self.N}, {self.K}]"
        
        if not dict_key in tailor_dict:
            print("Tailor")
            return opt_rlt, opt_split_k
        
        for i, value in enumerate(tailor_dict[dict_key]):
            opt_rlt.append(value[0])
            opt_split_k.append(value[1])
        return opt_rlt, opt_split_k
        
    def JIT(self, transpose_a=False, transpose_b=False, input_type="float", out_type="float",  rlt_json_dir = ""):
        template = BatchedGEMMTemplate()
        template_rlt = []
        split_template = 1
        
        m_shape = [int(m) for m in self.tuning_config["tbt_m"]]
        n_shape = [int(n) for n in self.tuning_config["tbt_n"]]
        k_shape = [8, 8, 8]
        kstage = [2, 2, 2] #8
        
        opt_rlt = self.create_cutlass_opt(m_shape=m_shape, n_shape=n_shape, k_shape=k_shape, kstage_shape=kstage, transpose_a=transpose_a, transpose_b=transpose_b)
        opt_split_k = []
        
        #create template
        tmp_function_body = ""
        tmp_function_body_arr = []
        for i in range(len(opt_rlt)):
            tmp_function_body += f"${{func_{i%split_template}}}\n\n"
            
            if (i+1) % split_template == 0:
                tmp_function_body_arr.append(tmp_function_body)
                tmp_function_body = ""
        if len(opt_rlt) % split_template != 0:
            tmp_function_body_arr.append(tmp_function_body)
        
        for i, value in enumerate(tmp_function_body_arr):
            function_body = {"function_body": value}
            template_rlt.append(template.substitute_template(template.template, function_body))
        
        #fill function body
        template_rlt_index = 0
        cutlass_function_body = {}
        for i, value in enumerate(opt_rlt):
            func_key = f"func_{i%split_template}"
            
            split_k = "split_k" if len(opt_split_k) == 0 else str(opt_split_k[i])
            
            cutlass_function_body[func_key] = template.cutlass_gemm_func(opt_rlt=value, number=i%split_template,
                                                                         transpose_a=transpose_a, transpose_b=transpose_b,
                                                                         rlt_json_dir=rlt_json_dir, opt_split_k=split_k, sm=self.sm)
            
            if (i+1) % split_template == 0:
                template_rlt[template_rlt_index] = template.substitute_template(template_rlt[template_rlt_index], cutlass_function_body)
                
                cutlass_function_body = {}
                template_rlt_index += 1
        
        if len(opt_rlt) % split_template != 0:
            template_rlt[template_rlt_index] = template.substitute_template(template_rlt[template_rlt_index], cutlass_function_body)
          
        #fill template lda, ldb, ldc, dtype
        for i, value in enumerate(template_rlt):
            opt_rlt_len = split_template if ((i+1) * split_template) < len(opt_rlt) else len(opt_rlt) - (i * split_template)
            template_rlt[i] = template.main_pre(template_rlt[i], transpose_a, transpose_b, input_type, out_type, opt_rlt_len)
        
        #make main execute body
        template_rlt_index = 0
        tmp_exec_main = ""
        for i in range(len(opt_rlt)):
            tmp_exec_main += f"${{exec_body_{i%split_template}}}\n\n\t"
            
            if (i + 1) % split_template == 0:
                exec_body_value = {"exec_body": tmp_exec_main}
                template_rlt[template_rlt_index] = template.substitute_template(template_rlt[template_rlt_index], exec_body_value)
                template_rlt_index += 1
                tmp_exec_main = ""
        if len(opt_rlt) % split_template != 0:
            exec_body_value = {"exec_body": tmp_exec_main}
            template_rlt[template_rlt_index] = template.substitute_template(template_rlt[template_rlt_index], exec_body_value)
            
        #fill template function call in main
        for i, value in enumerate(template_rlt):
            opt_rlt_len = split_template if ((i+1) * split_template) < len(opt_rlt) else len(opt_rlt) - (i * split_template)
            template_rlt[i] = template.main_exec(template_rlt[i], opt_rlt_len)
        
        return template_rlt
    
    def _compile(self, opt):
        compile_target = "nvcc"
        
        file_name = opt[0]
        object_file = opt[1]
        
        real_path = os.path.dirname(__file__)
        cutlass_path = f"{real_path}/../../../../3rdparty"
        
        #change when you use ohter computer
        compile_option = f"-O3 -arch=sm_{self.sm} --std=c++17 -I {cutlass_path}/cutlass/include -I {cutlass_path}/cutlass/examples -I {cutlass_path}/cutlass/tools/util/include"
        
        command_line = compile_target +  " " + file_name + " " + compile_option + " " + "-o " + object_file
        
        if not os.path.exists(object_file):
            print(command_line)
            os.system(command_line)
    
    def _codeGen(self, transpose_a=False, transpose_b=False, input_type="float", out_type="float"):
        # real_path = os.path.dirname(__file__)
        real_path = str(self.tuning_config["tmp_dir"])
        file_tailor = ""
        
        file_transpose = ""
        file_transpose += "T" if transpose_a == False else "N"
        file_transpose += "T" if transpose_b == False else "N"
        
        dir_path = f"{real_path}/src_cutlass_batch{file_tailor}"
        rlt_dir_path = f"{real_path}/rlt_cutlass_batch_{file_transpose}{file_tailor}"
        
        template_rlt = self.JIT(transpose_a=transpose_a, transpose_b=transpose_b, input_type=input_type, out_type=out_type, rlt_json_dir=rlt_dir_path)
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        if not os.path.exists(rlt_dir_path):
            os.makedirs(rlt_dir_path)
        
        file_name = []
        object_file = []
        pool_parameter = []
        
        for i in range(len(template_rlt)):
            single_file = f"{dir_path}/cutlass_batch_gemm_{file_transpose}_{i}{file_tailor}.cu"
            single_object_file = f"{dir_path}/cutlass_batch_gemm_{file_transpose}_{i}{file_tailor}" 
            
            file_name.append(single_file)
            object_file.append(single_object_file)
            pool_parameter.append([single_file, single_object_file])
            
            with open(file_name[i], "w") as f:
                f.write(template_rlt[i])
        
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        pool.map(self._compile, pool_parameter)
        
        return object_file, rlt_dir_path
    
    def _run(self, object_file=[], tailor_split_k=[]):
        
        if self.tailor == True:
            for i, value in enumerate(object_file):
                exec = f"{value} -b {self.batch} -m {self.M} -n {self.N} -k {self.K} -s {tailor_split_k[i] if tailor_split_k[i] > 0 else 1}"
                os.system(exec)
        else:            
            for i, value in enumerate(object_file):
                exec_naive = f"{value} -b {self.batch} -m {self.M} -n {self.N} -k {self.K} -s 1 -d {self.target_node}"
                os.system(exec_naive)
                for split_k in range(self.split_k[0], self.split_k[1] + self.split_k[2], self.split_k[2]):
                    exec = f"{value} -b {self.batch} -m {self.M} -n {self.N} -k {self.K} -s {split_k} -d {self.target_node}"
                    os.system(exec)
    
    def _codeGenTailor(self, transpose_a=False, transpose_b=False, input_type="float", out_type="float"):
        real_path = os.path.dirname(__file__)
        
        file_tailor = ""
        
        file_transpose = ""
        file_transpose += "T" if transpose_a == False else "N"
        file_transpose += "T" if transpose_b == False else "N"
        
        dir_path = f"{real_path}/src_cutlass_batch"
        rlt_dir_path = f"{real_path}/rlt_cutlass_batch_{file_transpose}{file_tailor}"
        
        candidate = {}
        
        with open(f"{real_path}/tailor_3090.json", "r") as file:
                for line in file:
                    json_line = json.loads(line)
                    
                    if str(json_line["gemm"]) not in candidate:
                        candidate.update({str(json_line["gemm"]): []})
                    
                    candidate[str(json_line["gemm"])].append([json_line["dim"], json_line["split_k"], json_line["swizzle"]])
        
        candidate_key = f"{[self.M * self.batch, self.N, self.K]}"
        
        m_shape = [32, 256, 32]
        n_shape = [32, 256, 32]
        k_shape = [8, 8, 8]
        kstage = [2, 2, 2] #8
        
        opt_rlt = self.create_cutlass_opt(m_shape=m_shape, n_shape=n_shape, k_shape=k_shape, kstage_shape=kstage, transpose_a=transpose_a, transpose_b=transpose_b)
        
        k_count = 0
        object_file = []
        split_k = []
        
        for candidate_param in candidate[candidate_key]:        
            param = candidate_param[0]
            
            try:
                k_count += 1
                x = opt_rlt.index(param)
                single_object_file = f"{dir_path}/cutlass_batch_gemm_{file_transpose}_{x}{file_tailor}"
                object_file.append(single_object_file)
                split_k.append(candidate_param[1])
            except ValueError:
                print("warp > 8")
            
            if k_count == 5:
                break
        
        if not os.path.exists(rlt_dir_path):
            os.makedirs(rlt_dir_path)
        
        return object_file, rlt_dir_path, split_k
        
    
    def eval_cutlassOracle(self, transpose_a=False, transpose_b=False, input_type="float", out_type="float"):
        if self.tailor == True:
            object_file, rlt_dir, split_k_param = self._codeGenTailor(transpose_a=transpose_a, transpose_b=transpose_b, input_type=input_type, out_type=out_type)
            rlt_json = f"{rlt_dir}/{self.batch}_{self.M}_{self.N}_{self.K}.json"
            if not os.path.exists(rlt_json):
                self._run(object_file=object_file, tailor_split_k=split_k_param)
        else:
            object_file, rlt_dir = self._codeGen(transpose_a=transpose_a, transpose_b=transpose_b, input_type=input_type, out_type=out_type)
            rlt_json = f"{rlt_dir}/{self.batch}_{self.M}_{self.N}_{self.K}.json"
            if not os.path.exists(rlt_json):
                self._run(object_file=object_file)
        
        rlt = []
        with open(rlt_json, "r") as f:
            for line in f:
                json_data = json.loads(line.strip())
                rlt.append(json_data)
                
        sorted_data = sorted(rlt, key=lambda x: x['time'])
        
        for i, value in enumerate(sorted_data):    
            if value["time"] != -1 and value["time"] != 0:
                fastest_cutlass_time = value["time"]
                fastest_cutlass_tile = value["dim"]
                fastest_cutltass_split = value["split_k"]
                
                # if fastest_cutltass_split > 1:
                #     continue
                
                break
        
        print(f"{self.batch}, {self.M}, {self.N}, {self.K}")
        print(f"{fastest_cutlass_tile}, {fastest_cutltass_split}")
        print(fastest_cutlass_time)
        
        return list(fastest_cutlass_tile), fastest_cutltass_split, 8
        
class GEMMTemplate:
    def __init__(self):
        self.template = """
#include<iostream>
#include<cuda_runtime.h>

#include <unistd.h>
#include<string>
#include<fstream>      

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"
#include <cutlass/util/host_tensor.h>

${function_body}

int main(int argc, char *argv[]){
    float *rlt = new float[${opt_rlt}];

    int M = 64;
    int N = 64;
    int K = 64;
    int Batch = 1;
    int split_k = 1;
    int device_id;
    
    int option;
    while((option = getopt(argc, argv, "m:n:k:b:s:d:")) != -1){
        switch(option){
            case 'm':
                M = std::stoi(optarg);
                break;
            case 'n':
                N = std::stoi(optarg);
                break;
            case 'k':
                K = std::stoi(optarg);
                break;
            case 'b':
                Batch = std::stoi(optarg);
                break;
            case 's':
                split_k = std::stoi(optarg);
                break;
            case 'd':
                device_id = std::stoi(optarg);
                break;    
            case '?':
                break;
        }
    }
    
    cudaSetDevice(device_id);
    
    int const lda = ${lda};
    int const ldb = ${ldb};
    int const ldc = ${ldc};
    
    int const count_A = Batch * M * K;
    int const count_B = Batch * N * K;
    int const count_C = Batch * M * N;
    
    long long int batch_stride_A = static_cast<long long int>(M) * static_cast<long long int>(K);
    long long int batch_stride_B = static_cast<long long int>(K) * static_cast<long long int>(N);
    long long int batch_stride_C = static_cast<long long int>(M) * static_cast<long long int>(N);
    
    ${outType_1} alpha = static_cast<${outType_1}>(1.0f);
    ${outType_2} beta = static_cast<${outType_2}>(0.0f);
    
    std::vector<${outType_1}> host_A(count_A, 1.2f);
    std::vector<${outType_1}> host_B(count_B, 1.0f);
    std::vector<${outType_2}> host_C(count_C);
    
    ${outType_1} *A;
    ${outType_1} *B;
    ${outType_2} *C;
    
    cudaMalloc(&A, count_A * sizeof(${outType_1}));
    cudaMalloc(&B, count_B * sizeof(${outType_1}));
    cudaMalloc(&C, count_C * sizeof(${outType_2}));
    
    cudaMemcpy(A, host_A.data(), count_A * sizeof(${outType_1}), cudaMemcpyHostToDevice);
    cudaMemcpy(B, host_B.data(), count_B * sizeof(${outType_1}), cudaMemcpyHostToDevice);
    cudaMemcpy(C, host_C.data(), count_C * sizeof(${outType_2}), cudaMemcpyHostToDevice);
    
    //warmp up
    for(int i = 0; i < 2; i++){
        cutlass_strided_bathed_sgemm_0(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, 1, 1);
    }
    
    ${exec_body}
    
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    
    return 0;
}
        """
        
        self.function_template = """
float cutlass_strided_bathed_sgemm_${number}(
    int m, int n, int k,
    ${outType_1} alpha, ${outType_1} const *A, int lda, long long int batch_stride_A,
    ${outType_1} const *B, int ldb, long long int batch_stride_B,
    ${outType_2} *C, int ldc, long long int batch_stride_C,
    ${outType_2} beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::Gemm<
                                                    ${outType_1}, ${layout_0},
                                                    ${outType_1}, ${layout_1},
                                                    ${outType_2}, ${layout_2},
                                                    ${outType_2},
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm${sm},
                                                    cutlass::gemm::GemmShape<${MM}, ${MN}, ${MK}>,
                                                    cutlass::gemm::GemmShape<${WM}, ${WN}, ${WK}>,
                                                    cutlass::gemm::GemmShape<1, 1, 1>,
                                                    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
                                                    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<${sizzle_option}>,
                                                    ${Stage},
                                                    ${Al},
                                                    ${Al},
                                                    true,
                                                    cutlass::arch::OpMultiplyAdd
                                                    >;
    
    Gemm gemm_op;
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    Gemm::Arguments arguments{
            {m, n, k},
            {A, lda},
            {B, ldb},
            {C, ldc},
            {C, ldc},
            {alpha, beta},
            ${split_k}
        };
        
        int *workspace;
        int workspace_size = ((m + (${MM} - 1)) / ${MM}) * ((m + (${MN} - 1)) / ${MN}) * sizeof(int);
        cudaMalloc(&workspace, workspace_size);
        cudaMemset(workspace, 0, workspace_size);
        
        cutlass::Status status = gemm_op.can_implement(arguments);
        if(status != cutlass::Status::kSuccess) return -1;
        
        status = gemm_op.initialize(arguments, workspace);
        if(status != cutlass::Status::kSuccess) return -1;
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < ${repeat}; i++){
        status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "${rlt_json_dir}/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\\"dim\\": [[${MM}, ${MN}, ${MK}], [${WM}, ${WN}, ${WK}], [${Stage}], [${Al}]], \\"split_k\\": " + std::to_string(split_k)
                                    + ", \\"sizzle\\": ${sizzle_option}" + " , \\"time\\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }
            cudaFree(workspace);
            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);
    
    cudaFree(workspace);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "${rlt_json_dir}/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\\"dim\\": [[${MM}, ${MN}, ${MK}], [${WM}, ${WN}, ${WK}], [${Stage}], [${Al}]], \\"split_k\\": " + std::to_string(split_k)
                            + ", \\"sizzle\\": ${sizzle_option}" + " ,\\"time\\": " + std::to_string(total_time/${repeat}) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / ${repeat};
    
}
        """
    
    def cutlass_gemm_func(self, opt_rlt=[], number=0, repeat=20, transpose_a=False, transpose_b=True, rlt_json_dir="", opt_split_k="split_k", sm=70):
        shared_mem = opt_rlt[0]
        reg_mem = opt_rlt[1]
        stages = opt_rlt[2][0]
        alignment = opt_rlt[3][0]
        sizzle = opt_rlt[4][0]
        
        template_value = {}
        template_value["MM"] = str(shared_mem[0])
        template_value["MN"] = str(shared_mem[1])
        template_value["MK"] = str(shared_mem[2])
        
        template_value["WM"] = str(reg_mem[0])
        template_value["WN"] = str(reg_mem[1])
        template_value["WK"] = str(reg_mem[2])
        
        template_value["Stage"] = str(stages)
        template_value["Al"] = str(alignment)
        template_value["repeat"] = str(repeat)
        template_value["number"] = str(number)
        
        template_value["layout_0"] = "cutlass::layout::ColumnMajor" if transpose_a == True else "cutlass::layout::RowMajor"
        template_value["layout_1"] = "cutlass::layout::ColumnMajor" if transpose_b == True else "cutlass::layout::RowMajor"
        template_value["layout_2"] = "cutlass::layout::RowMajor"
        template_value["rlt_json_dir"] = rlt_json_dir
        
        template_value['split_k'] = opt_split_k
        
        if int(sm) == 89:
            template_value["sm"] = str(86)
        else:
            template_value["sm"] = str(sm)
        
        template_value["sizzle_option"] = str(sizzle)
        
        template = self.substitute_template(self.function_template, template_value)
        return template
    
    def main_pre(self, template, transpose_a=False, transpose_b=False, dtype="float", outtype="float", len=0):
        value = {}
        
        
        value["lda"] = "K" if transpose_a == False else "M"
        value["ldb"] = "N" if transpose_b == False else "K"
        value["ldc"] = "N"
        
        value["outType_1"] = dtype
        value["outType_2"] = outtype
        
        value["opt_rlt"] = str(len)
        
        template = self.substitute_template(template, value)
              
        return template
    
    def main_exec(self, template, len=0):
        value = {}
        for i in range(len):
            value_key = f"exec_body_{str(i)}"
            value[value_key] = f"rlt[{str(i)}] = cutlass_strided_bathed_sgemm_{str(i)}(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);"
            
        template = self.substitute_template(template, value)
        return template
    
    def split_k_exec(self, template, split_k=1):
        value = {'split_k': str(split_k)}
        template = self.substitute_template(template, value)
        return template
    
    def substitute_template(self, template, values):
        text = template
        changed = True
        while changed:
            changed = False
            for key, value in values.items():
                regex = "\\$\\{%s\\}" % key
                newtext = re.sub(regex, value, text)
                if newtext != text:
                    changed = True
                text = newtext
        return text
    
class ProfileGemm:
    def __init__(self, batch=1, M=64, N=64, K=64, tuning_config={}):
        self.batch = int(batch)
        self.M = int(M)
        self.N = int(N)
        self.K = int(K)
        self.tuning_config = tuning_config
        self.split_k = [int(split_k) for split_k in tuning_config["split_k_range"]]
        self.tailor = False
        self.sm = int(self.tuning_config["sm"])
        # self.target_node = int(tuning_config["target_node"])
        self.target_node = 0
    
    def create_cutlass_opt(self, m_shape=[], n_shape=[], k_shape=[], kstage_shape=[], sizzle_stage=[], alignment=1, transpose_a=False, transpose_b=False):
    # #_shape = [start, end, stride]
    # return option
        compile_option = []
        
        def iswarp(shared_m, shared_n, warp_m, warp_n, kstage=2):
            if((shared_m / warp_m) * (shared_n / warp_n)) > 16:
                return False
            
            if (shared_m % warp_m) != 0 or (shared_n % warp_n) != 0:
                return False
            
            tmp_warp = (shared_m // warp_m) * (shared_n // warp_n) * 32
            
            if tmp_warp > 256:
                return False
            
            if transpose_a == True:
                if tmp_warp % shared_m != 0 or (shared_m % tmp_warp != 0 and tmp_warp < shared_m):
                    return False
            
            if transpose_b == False:
                if tmp_warp % shared_n != 0 or (shared_n % tmp_warp and tmp_warp < shared_n) != 0:
                    return False
            
            warpNumThreadsM = 8
            
            if warp_m <= warp_n:
                warpNumThreadsM = 4
                
            if kstage > 2:
                warpNumThreadsM = 4
                warpNumThreadsN = 8
            
            warpNumThreadsN = 32 / warpNumThreadsM
            
            ThreadTileM = warp_m / warpNumThreadsM
            ThreadTileN = warp_n / warpNumThreadsN
            
            if warp_m % warpNumThreadsM != 0 or warp_n % warpNumThreadsN != 0:
                return False
            
            laneM = min(4, ThreadTileM)
            laneN = min(4, ThreadTileN)
            
            paddingM = 8
            paddingN = 8
            
            if transpose_a == True:
                paddingM = 0
            if transpose_b == False:
                paddingN = 0
            
            if paddingM % laneM != 0 or paddingN % laneN != 0:
                return False
            
            if warp_m / warpNumThreadsM < 0 or warp_n / warpNumThreadsN < 0:
                return False
            
            if warp_m % warpNumThreadsM != 0 or warp_n % warpNumThreadsN != 0:
                return False
            
            iterationA = warp_m / warpNumThreadsM
            iterationB = warp_n / warpNumThreadsN
        
            if iterationA % laneM != 0 or iterationB % laneN != 0:
                return False
            
            return True
        
        for stage in range(kstage_shape[0], kstage_shape[1] + kstage_shape[2], kstage_shape[2]):
            for shared_m in range(m_shape[0], m_shape[1] + m_shape[2], m_shape[2]):
                for shared_n in range(n_shape[0], n_shape[1] + n_shape[2], n_shape[2]):
                    for shared_k in range(k_shape[0], k_shape[1] + k_shape[2], k_shape[2]):
                        for warp_m in range(1, shared_m + 1):
                            for warp_n in range(1, shared_n + 1):
                                if iswarp(shared_m, shared_n, warp_m, warp_n, stage) is False:
                                    continue
                                
                                for sizzle in sizzle_stage:
                                    op = [[shared_m, shared_n, shared_k], [warp_m , warp_n, shared_k], [stage], [alignment], [sizzle]]
                                    compile_option.append(op)
        
        return compile_option
    
    def create_tailor_opt(self):
        tailor_dict = {}
        
        real_path = os.path.dirname(__file__)
        with open(f"{real_path}/tailor.txt", "r") as f:
            for line in f:
                line = line.split(' ')
                dict_key = f"{line[6]} {line[7]} {line[8]}"
                
                if len(line[9]) < 4:
                    continue
                
                parameter = line[9].split('_')
                parameter_block = parameter[0].split('x')
                parameter_block = [int(parameter_block[0]), int(parameter_block[1]), 8]
                
                parameter_warp = parameter[1].split("x")
                parameter_warp = [int(parameter_warp[0]), int(parameter_warp[1]), 8]
                
                split_k = parameter[2]
                split_k = split_k.replace("\n", "")
                split_k = int(split_k) if int(split_k) > 0 else 1
                
                if not dict_key in tailor_dict:
                    tailor_dict.update({dict_key: []})
                tailor_dict[dict_key].append([[parameter_block, parameter_warp, [2], [1]], split_k])
                # print(f"{dict_key}: {parameter_block}, {parameter_warp}, {split_k}")
        
        opt_rlt = []
        opt_split_k = []
        dict_key = f"{self.M}, {self.batch * self.N}, {self.K}]"
        
        if not dict_key in tailor_dict:
            print("NONe")
            return opt_rlt, opt_split_k
        
        for i, value in enumerate(tailor_dict[dict_key]):
            opt_rlt.append(value[0])
            opt_split_k.append(value[1])
        return opt_rlt, opt_split_k
        
    def JIT(self, transpose_a=False, transpose_b=False, input_type="float", out_type="float",  rlt_json_dir = ""):
        template = GEMMTemplate()
        template_rlt = []
        split_template = 1
        
        m_shape = [int(m) for m in self.tuning_config["tbt_m"]]
        n_shape = [int(n) for n in self.tuning_config["tbt_n"]]
        k_shape = [8, 8, 8]
        kstage = [2, 2, 2] #8
        sizzle_stage = [int(s) for s in self.tuning_config["swizzle_range"]]
        
        opt_rlt = self.create_cutlass_opt(m_shape=m_shape, n_shape=n_shape, k_shape=k_shape, kstage_shape=kstage, sizzle_stage=sizzle_stage, transpose_a=transpose_a, transpose_b=transpose_b)
        opt_split_k = []
        
        #create template
        tmp_function_body = ""
        tmp_function_body_arr = []
        for i in range(len(opt_rlt)):
            tmp_function_body += f"${{func_{i%split_template}}}\n\n"
            
            if (i+1) % split_template == 0:
                tmp_function_body_arr.append(tmp_function_body)
                tmp_function_body = ""
        if len(opt_rlt) % split_template != 0:
            tmp_function_body_arr.append(tmp_function_body)
        
        for i, value in enumerate(tmp_function_body_arr):
            function_body = {"function_body": value}
            template_rlt.append(template.substitute_template(template.template, function_body))
        
        #fill function body
        template_rlt_index = 0
        cutlass_function_body = {}
        for i, value in enumerate(opt_rlt):
            func_key = f"func_{i%split_template}"
            
            split_k = "split_k" if len(opt_split_k) == 0 else str(opt_split_k[i])
            
            cutlass_function_body[func_key] = template.cutlass_gemm_func(opt_rlt=value, number=i%split_template,
                                                                         transpose_a=transpose_a, transpose_b=transpose_b,
                                                                         rlt_json_dir=rlt_json_dir, opt_split_k=split_k, sm=self.sm)
            
            if (i+1) % split_template == 0:
                template_rlt[template_rlt_index] = template.substitute_template(template_rlt[template_rlt_index], cutlass_function_body)
                
                cutlass_function_body = {}
                template_rlt_index += 1
        
        if len(opt_rlt) % split_template != 0:
            template_rlt[template_rlt_index] = template.substitute_template(template_rlt[template_rlt_index], cutlass_function_body)
          
        #fill template lda, ldb, ldc, dtype
        for i, value in enumerate(template_rlt):
            opt_rlt_len = split_template if ((i+1) * split_template) < len(opt_rlt) else len(opt_rlt) - (i * split_template)
            template_rlt[i] = template.main_pre(template_rlt[i], transpose_a, transpose_b, input_type, out_type, opt_rlt_len)
        
        #make main execute body
        template_rlt_index = 0
        tmp_exec_main = ""
        for i in range(len(opt_rlt)):
            tmp_exec_main += f"${{exec_body_{i%split_template}}}\n\n\t"
            
            if (i + 1) % split_template == 0:
                exec_body_value = {"exec_body": tmp_exec_main}
                template_rlt[template_rlt_index] = template.substitute_template(template_rlt[template_rlt_index], exec_body_value)
                template_rlt_index += 1
                tmp_exec_main = ""
        if len(opt_rlt) % split_template != 0:
            exec_body_value = {"exec_body": tmp_exec_main}
            template_rlt[template_rlt_index] = template.substitute_template(template_rlt[template_rlt_index], exec_body_value)
            
        #fill template function call in main
        for i, value in enumerate(template_rlt):
            opt_rlt_len = split_template if ((i+1) * split_template) < len(opt_rlt) else len(opt_rlt) - (i * split_template)
            template_rlt[i] = template.main_exec(template_rlt[i], opt_rlt_len)
        
        return template_rlt
    
    def _compile(self, opt):
        compile_target = "nvcc"
        
        file_name = opt[0]
        object_file = opt[1]
        
        real_path = os.path.dirname(__file__)
        cutlass_path = f"{real_path}/../../../../3rdparty"
        
        #change when you use ohter computer
        compile_option = f"-O3 -arch=sm_{self.sm} --std=c++17 -I {cutlass_path}/cutlass/include -I {cutlass_path}/cutlass/examples -I {cutlass_path}/cutlass/tools/util/include"
        
        command_line = compile_target +  " " + file_name + " " + compile_option + " " + "-o " + object_file
        
        if not os.path.exists(object_file):
            print(command_line)
            os.system(command_line)
    
    def _codeGen(self, transpose_a=False, transpose_b=False, input_type="float", out_type="float"):
        # real_path = os.path.dirname(__file__)
        real_path = str(self.tuning_config["tmp_dir"])
        file_tailor = "" if self.tailor == False else "_Tailor"
        
        file_transpose = ""
        file_transpose += "T" if transpose_a == False else "N"
        file_transpose += "T" if transpose_b == False else "N"
        
        dir_path = f"{real_path}/src_cutlass{file_tailor}"
        rlt_dir_path = f"{real_path}/rlt_cutlass_{file_transpose}{file_tailor}"
        
        template_rlt = self.JIT(transpose_a=transpose_a, transpose_b=transpose_b, input_type=input_type, out_type=out_type, rlt_json_dir=rlt_dir_path)
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        if not os.path.exists(rlt_dir_path):
            os.makedirs(rlt_dir_path)
        
        file_name = []
        object_file = []
        
        pool_parameter = []
        
        for i in range(len(template_rlt)):
            single_file = f"{dir_path}/cutlass_gemm_{file_transpose}_{i}{file_tailor}.cu"
            single_object_file = f"{dir_path}/cutlass_gemm_{file_transpose}_{i}{file_tailor}"
            
            file_name.append(single_file)
            object_file.append(single_object_file)
            
            with open(file_name[i], "w") as f:
                f.write(template_rlt[i])
            
            pool_parameter.append([single_file, single_object_file])
        
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        pool.map(self._compile, pool_parameter)
        
        return object_file, rlt_dir_path
    
    def _run(self, object_file=[], tailor_split_k=[]):
        
        if self.tailor == True:
            for i, value in enumerate(object_file):
                exec = f"{value} -b {self.batch} -m {self.M} -n {self.N} -k {self.K} -s {tailor_split_k[i] if tailor_split_k[i] > 0 else 1}"
                os.system(exec)
        else:
            for i, value in enumerate(object_file):
                exec_naive = f"{value} -b {self.batch} -m {self.M} -n {self.N} -k {self.K} -s 1 -d {self.target_node}"
                os.system(exec_naive)
                
                for split_k in range(self.split_k[0], self.split_k[1] + self.split_k[2], self.split_k[2]):
                    exec = f"{value} -b {self.batch} -m {self.M} -n {self.N} -k {self.K} -s {split_k} -d {self.target_node}"
                    os.system(exec)
    
    def _codeGenTailor(self, transpose_a=False, transpose_b=False, input_type="float", out_type="float"):
        real_path = os.path.dirname(__file__)
        
        file_tailor = ""
        
        file_transpose = ""
        file_transpose += "T" if transpose_a == False else "N"
        file_transpose += "T" if transpose_b == False else "N"
        
        dir_path = f"{real_path}/src_cutlass"
        rlt_dir_path = f"{real_path}/rlt_cutlass_{file_transpose}{file_tailor}"
        
        candidate = {}
        
        with open(f"{real_path}/tailor_3090.json", "r") as file:
                for line in file:
                    json_line = json.loads(line)
                    
                    if str(json_line["gemm"]) not in candidate:
                        candidate.update({str(json_line["gemm"]): []})
                    
                    candidate[str(json_line["gemm"])].append([json_line["dim"], json_line["split_k"], json_line["swizzle"]])
        
        candidate_key = f"{[self.M * self.batch, self.N, self.K]}"
        
        m_shape = [32, 256, 32]
        n_shape = [32, 256, 32]
        k_shape = [8, 8, 8]
        kstage = [2, 2, 2] #8
        sizzle_stage = [8, 4, 2, 1]
        
        opt_rlt = self.create_cutlass_opt(m_shape=m_shape, n_shape=n_shape, k_shape=k_shape, kstage_shape=kstage, sizzle_stage=sizzle_stage, transpose_a=transpose_a, transpose_b=transpose_b)
        
        k_count = 0
        object_file = []
        split_k = []
        
        for candidate_param in candidate[candidate_key]:
            
            try:
                k_count += 1
                param = [candidate_param[0][0], candidate_param[0][1], candidate_param[0][2], candidate_param[0][3], [candidate_param[2]]]
                x = opt_rlt.index(param)
                single_object_file = f"{dir_path}/cutlass_gemm_{file_transpose}_{x}{file_tailor}"
                object_file.append(single_object_file)
                split_k.append(candidate_param[1])
            except ValueError:
                print("warp > 8")
            
            if k_count == 5:
                break
        
        if not os.path.exists(rlt_dir_path):
            os.makedirs(rlt_dir_path)
        
        return object_file, rlt_dir_path, split_k
    
    def eval_cutlassOracle(self, transpose_a=False, transpose_b=False, input_type="float", out_type="float"):
        
        if self.tailor == True:
            object_file, rlt_dir, split_k_param = self._codeGenTailor(transpose_a=transpose_a, transpose_b=transpose_b, input_type=input_type, out_type=out_type)
            rlt_json = f"{rlt_dir}/{self.batch}_{self.M}_{self.N}_{self.K}.json"
            if (not os.path.exists(rlt_json)): #(self.M, self.N, self.K) == (4096, 32000, 3200)
                self._run(object_file=object_file, tailor_split_k=split_k_param)
        else:
            object_file, rlt_dir = self._codeGen(transpose_a=transpose_a, transpose_b=transpose_b, input_type=input_type, out_type=out_type)
            rlt_json = f"{rlt_dir}/{self.batch}_{self.M}_{self.N}_{self.K}.json"
            if (not os.path.exists(rlt_json)): #  or (self.M, self.N, self.K) == (2048, 128256, 4096)
                self._run(object_file=object_file)
        
        rlt = []
        with open(rlt_json, "r") as f:
            for line in f:
                json_data = json.loads(line.strip())
                rlt.append(json_data)
                
        sorted_data = sorted(rlt, key=lambda x: x['time'])
        
        for i, value in enumerate(sorted_data):    
            if value["time"] != -1 and value["time"] != 0:                
                fastest_cutlass_time = value["time"]
                fastest_cutlass_tile = value["dim"]
                fastest_cutltass_split = value["split_k"]
                fastest_cutlass_sizzle = value["sizzle"]
                
                # if fastest_cutltass_split > 1:
                #     continue
                
                break
        
        print(f"{self.M}, {self.N}, {self.K}")
        print(f"{fastest_cutlass_tile}, {fastest_cutltass_split}, {fastest_cutlass_sizzle}")
        print(fastest_cutlass_time)
        
        return list(fastest_cutlass_tile), int(fastest_cutltass_split), int(fastest_cutlass_sizzle)
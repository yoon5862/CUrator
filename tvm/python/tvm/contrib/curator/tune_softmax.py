from .library import *
import multiprocessing
import os
import json

def instantiate_softmax_template(attrs, func_args):
    template = """
    //cutlass support
    using ElementInputA = ${ElementInputA};
    const int batch = ${batch};
    const int M = ${M};
    const int N = ${N};
    
    void* ptr_a = (void*)(${arg0}->data);
    void* ptr_out = (void*)(out0->data);
    
    ${cutlass_op_name}<ElementInputA, ${row_per_access}, ${pack_size}, ${col_per_thread}, ${warp_count}>(static_cast<ElementInputA*>(ptr_a), static_cast<ElementInputA*>(ptr_out), batch, M, N);
    """
    
    aux_map = {}
    aux_map.update({"ElementInputA": attrs["ElementInputA"], "batch": attrs["batch"]})
    
    template = substitute_template(template, aux_map)
    
    for i, arg in enumerate(func_args):
        # print(type(arg))
        attrs["arg{}".format(i)] = arg
    
    return substitute_template(template, attrs)



class SoftmaxTemplate:
    def __init__(self,) -> None:
        self.template = """
            #include<iostream>
            #include<vector>
            #include<cuda_runtime.h>
            #include <unistd.h>
            #include<string>
            #include<fstream>
            
            #include"support/softmax.cuh"
            
            int main(int argc, char *argv[]){
                int M = 64;
                int N = 64;
                int Batch = 1;
                
                int option;
                while((option = getopt(argc, argv, "m:n:k:b:s:")) != -1){
                    switch(option){
                        case 'm':
                            M = std::stoi(optarg);
                            break;
                        case 'n':
                            N = std::stoi(optarg);
                            break;
                        case 'b':
                            Batch = std::stoi(optarg);
                            break;
                        case '?':
                            break;
                    }
                }
                
                int const count = M * N * Batch;
                
                std::vector<${in_type}> input(count, 0.001);
                std::vector<${out_type}> output(count, 0);
                                
                ${in_type} *d_in;
                ${out_type} *d_out;
                
                cudaMalloc(&d_in, count * sizeof(${in_type}));
                cudaMalloc(&d_out, count * sizeof(${out_type}));
                
                cudaMemcpy(d_in, input.data(), count * sizeof(${in_type}), cudaMemcpyHostToDevice);
                cudaMemcpy(d_out, output.data(), count * sizeof(${out_type}), cudaMemcpyHostToDevice);
                
                cudaEvent_t start, end;
                cudaEventCreate(&start);
                cudaEventCreate(&end);
                cudaDeviceSynchronize();
                
                bool can_implement = cutlass::support::can_implement<${pack_size}, ${col_per_thread}, ${warp_count}>(Batch, M, N);
                if(can_implement == false){
                    std::fstream dataFile;
                    std::string fileName = "${rlt_json_dir}/" + std::to_string(Batch) + "_" +
                                            std::to_string(M) + "_" + std::to_string(N) + ".json";
                                            
                    std::string json = "{\\"dim\\": [${row_per_access}, ${pack_size}, ${col_per_thread}, ${warp_count}], \\"time\\": " + std::to_string(-1) + "}";
                    
                    dataFile.open(fileName, std::ios::app);
                    dataFile << json << std::endl;
                    
                    return 0;
                }
                
                //warmming up gpu
                for(int i = 0; i < 100; i++) cutlass::support::softmaxWarp<${in_type}, ${row_per_access}, ${pack_size}, ${col_per_thread}, ${warp_count}>(d_in, d_out, Batch, M, N);
                
                cudaEventRecord(start);
                cutlass::support::softmaxWarp<${in_type}, ${row_per_access}, ${pack_size}, ${col_per_thread}, ${warp_count}>(d_in, d_out, Batch, M, N);
                cudaEventRecord(end);
                cudaEventSynchronize(end);

                float time;
                cudaEventElapsedTime(&time, start, end);
                
                std::fstream dataFile_rlt;
                std::string fileName_rlt = "${rlt_json_dir}/" + std::to_string(Batch) + "_" +
                                        std::to_string(M) + "_" + std::to_string(N) + ".json";
                                        
                std::string json_rlt = "{\\"dim\\": [${row_per_access}, ${pack_size}, ${col_per_thread}, ${warp_count}], \\"time\\": " + std::to_string(time) + "}";
                
                dataFile_rlt.open(fileName_rlt, std::ios::app);
                dataFile_rlt << json_rlt << std::endl;
                
            }
            
        """
    
    def instantiate_softmax_template(self, opt, input_type="float", out_type="float", rlt_dir=""):
        value = {}
        template = self.template
        
        value["in_type"] = input_type
        value["out_type"] = out_type
        value["rlt_json_dir"] = rlt_dir
        
        value["row_per_access"] = str(opt[0])
        value["pack_size"] = str(opt[1])
        value["col_per_thread"] = str(opt[2])
        value["warp_count"] = str(opt[3])
        
        return substitute_template(template, value)
        
    
    
class ProfileSoftamx:
    def __init__(self, sm=86, path="./cutlass"):
        self.cache = {}
        self.sm= sm
        
        self.path = path
        self.real_path = os.path.dirname(__file__)
        self.cutlass_path = f"{self.real_path}/../../../../3rdparty/cutlass/include"
        
        self.profile_dir = f"{self.path}/src_cutlass_sofmax"
        self.rlt_dir = f"{self.path}/rlt_cutlass_softmax"
        
        if not os.path.exists(self.profile_dir):
            os.makedirs(self.profile_dir)
        if not os.path.exists(self.rlt_dir):
            os.makedirs(self.rlt_dir)
    
    def create_cutlass_opt(self, row_per_thread=[], pack_size=[], col_per_thread=[], warp_count = []):
        def valid(vectorization, thread_column):
            return thread_column % vectorization == 0
        
        opt = []
        for _row_per_thread in row_per_thread:
            for _pack_size in pack_size:
                for _col_per_thread in col_per_thread:
                    for _warp_count in warp_count:
                        if valid(_pack_size, _col_per_thread):
                            opt.append([_row_per_thread, _pack_size, _col_per_thread, _warp_count])
        
        return opt
    
    def JIT(self, input_type="float", out_type="float"):
        #create template
        row_per_thread = [1, 2, 3, 4]
        pack_size = [1, 2, 4]
        col_per_thread = [1, 2, 4, 8, 16, 32]
        warp_count = [2, 4, 8]
        
        opt = self.create_cutlass_opt(row_per_thread, pack_size, col_per_thread, warp_count)
        
        rlt_template = []
        template = SoftmaxTemplate()
        
        for index, value in enumerate(opt):
            rlt = template.instantiate_softmax_template(value, input_type, out_type, self.rlt_dir)
            rlt_template.append(rlt)
        
        return rlt_template
    
    def compile(self, opt):
        compile_target = "nvcc"
        
        file_name = opt[0]
        object_file = opt[1]
        
        compile_option = f"-O3 -arch=sm_{self.sm} --std=c++17 -I {self.cutlass_path}"
        
        command_line = compile_target + " " + file_name + " "  + compile_option + " " + "-o " + object_file
        
        if not os.path.exists(object_file):
            print(command_line)
            os.system(command_line)
        
    
    #return object file name
    def create_template(self, input_type="float", out_type="float"):
        template = self.JIT(input_type, out_type)
        assert len(template) > 0, "template doesn't create"
        
        file_name = []
        object_file = []
        
        pool_parameter = []
        
        for i, value in enumerate(template):
            single_file = f"{self.profile_dir}/cutlass_softamx_{i}.cu"
            single_object_file = f"{self.profile_dir}/cutlass_softamx_{i}"
            
            file_name.append(single_file)
            object_file.append(single_object_file)
            
            with open(file_name[i], "w") as f:
                f.write(value)
            
            pool_parameter.append([single_file, single_object_file])
        
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        pool.map(self.compile, pool_parameter)
        
        return object_file
    
    def run(self, object_file=[]):
        for i, value in enumerate(object_file):
            exec = f"{value} -b {self.batch} -m {self.M} -n {self.N}"
            os.system(exec)  
    
    def profile_oracle(self, input_type="float", out_type="float", batch=1, M=512, N=512):
        self.batch = batch
        self.M = M
        self.N = N
        
        if (self.batch, self.M, self.N) in self.cache:
            tiling = self.cache[(self.batch, self.M, self.N)]
            return tiling
        
        object_file = self.create_template(input_type=input_type, out_type=out_type)
        rlt_json = f"{self.rlt_dir}/{self.batch}_{self.M}_{self.N}.json"
        
        if not os.path.exists(rlt_json):
            self.run(object_file)
        
        rlt = []
        with open(rlt_json, "r") as f:
            for line in f:
                json_data = json.loads(line.strip())
                rlt.append(json_data)
        
        sorted_data = sorted(rlt, key=lambda x: x['time'])
        
        for i, value in enumerate(sorted_data):
            if value['time'] != -1 and value['time'] != 0:
                fastest_cutlass_time = value['time']
                fastest_cutlass_tile = value['dim']
                break
            
        print(f"{self.batch}, {self.M}, {self.N}")
        print(f"{fastest_cutlass_tile}")
        print(f"{fastest_cutlass_time}")
        
        self.cache[(self.batch, self.M, self.N)] = list(fastest_cutlass_tile)
        
        return list(fastest_cutlass_tile)
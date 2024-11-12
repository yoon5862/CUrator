/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Unit tests for thread-level GEMM
*/

#include "../../common/cutlass_unit_test.h"

#include "cutlass/layout/layout.h"
#include "cutlass/epilogue/thread/activation.h"

#include "cutlass/util/host_tensor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int N, typename Func>
__global__ void test_Epilogue_thread_activation(T *out, T *in) {

  cutlass::Array<T, N> *vec_out = reinterpret_cast<cutlass::Array<T, N> *>(out);
  cutlass::Array<T, N> *vec_in = reinterpret_cast<cutlass::Array<T, N> *>(in);

  Func func;
  vec_out[threadIdx.x] = func(vec_in[threadIdx.x]);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

//
// Reference
//

static double GELU_golden_input[] = {
    1.587425827980,  1.157652974129,  0.750432848930, -0.965980410576,
    -0.388184845448,  0.014422321692,  0.353164494038,  1.354383468628,
     0.167588576674,  0.272798538208, -0.377032428980,  1.923444747925,
     0.308164477348, -0.341318070889,  0.278338819742, -0.292668998241,
    -1.051743745804, -0.814175724983,  0.112737402320,  1.262938618660,
    -1.582363605499,  0.722016870975,  1.053453564644, -0.659764587879,
     0.734917521477,  0.091274201870,  0.604461073875, -0.219043627381,
    -0.136795744300,  0.960650205612, -1.805408835411,  0.091029644012,
    -1.023343324661,  0.147713735700, -0.499895423651,  1.351878166199,
    -1.631091356277, -0.336171895266, -1.612408638000,  0.090832948685,
    -0.658132910728, -0.326727777719, -1.986387014389,  0.787685871124,
    -1.015677452087, -0.225094825029,  0.876752018929,  0.744826257229,
     0.870290279388, -0.757595360279,  1.510331749916,  0.750012576580,
     0.906444966793, -0.915759027004,  1.260277032852, -0.158465340734,
    -0.109191477299, -0.817102134228,  0.391305118799, -0.524910449982,
     0.351349592209,  0.801979541779,  0.446691334248, -0.741077482700,
     1.205966711044, -0.910210072994,  0.945986449718,  0.784096539021,
     1.670521497726,  0.344931513071, -0.301411420107,  0.309870749712,
    -0.879704594612, -1.951189517975, -0.805817663670, -0.661812782288,
    -0.505914270878, -1.836273789406, -0.381845980883, -0.554707705975,
    -0.375447630882, -0.516645610332,  0.509586095810,  1.087131023407,
     2.664817094803, -1.558295488358, -0.076461032033, -0.504621028900,
     1.327111959457, -1.819981694221,  1.350415468216, -2.074112653732,
     1.501431345940, -1.339013576508,  0.162817999721, -1.473457217216,
     0.357770472765,  0.188413277268,  1.601302266121, -0.653882205486,
     0.856162548065,  0.763102591038, -0.526283502579,  0.581961452961,
     0.089969776571,  1.968745589256,  0.545802056789, -1.168786048889,
     1.206663012505, -0.109096683562, -1.223938226700,  0.744599223137,
    -1.779406785965,  0.766436159611, -0.579044401646, -1.002057313919,
    -0.715845823288, -0.562508940697,  0.886768460274,  2.327786445618,
    -0.148763969541, -0.918884515762, -0.367678701878, -1.105021238327,
    -0.461237311363,  0.158228352666, -0.254040330648,  1.427477598190,
     0.277530491352,  0.046293262392, -0.535557329655, -1.486695051193,
    -0.953706681728, -1.040495038033, -0.314667612314,  0.348172843456,
     0.522773325443,  0.025960063562, -0.482472360134,  1.993084549904,
    -0.253064930439, -0.012146313675, -2.166327714920,  0.398040622473,
    -0.022238900885, -0.443580865860, -0.898376941681, -0.571689844131,
     1.666979670525, -0.831176340580, -0.671057403088,  0.481970995665,
    -1.096243023872, -1.493894338608,  0.596651911736, -0.229505166411,
     1.165976166725,  0.905094027519,  0.049716457725, -1.362933635712,
    -0.366948783398,  1.461613893509, -0.718411505222,  0.895385026932,
    -0.763122260571,  1.329716682434,  1.366570711136, -0.086544901133,
     0.059739742428,  0.940766513348, -0.272854357958, -1.738811373711,
    -0.361239165068,  0.696977972984,  1.288442254066,  1.264815807343,
    -0.573566436768, -1.141678214073,  0.081865988672, -0.886228799820,
    -0.236933603883,  1.050115466118, -0.538952171803,  0.651773929596,
    -0.220034509897, -1.198960781097,  1.247478365898, -0.053529661149,
     0.639809548855,  1.672434806824,  0.511088073254, -1.179364681244,
    -0.730427742004,  0.157630980015,  0.389369845390, -0.925578773022,
    -0.093250080943, -0.391062080860,  0.852983593941,  1.868778109550,
    -1.198786258698,  0.604997038841, -1.482687234879, -2.469333171844,
     0.718807697296, -0.559609353542,  2.187228441238, -2.927527904510,
     0.148535788059, -0.097280368209,  0.674131810665, -1.137645959854,
     0.792729616165, -1.166317462921, -0.498791724443,  1.675866723061,
    -0.137909621000, -0.653263568878, -2.281216144562,  0.296096831560,
     2.002410173416,  1.083609819412,  0.933580815792, -1.504760265350,
     2.185185909271,  0.286121010780, -1.035485863686, -0.216372340918,
    -0.274334043264, -0.849510788918, -1.397169828415, -0.407644748688,
     0.159476816654, -0.170650705695,  0.335193097591, -0.156852483749,
     0.036168430001,  0.858105242252, -1.086121797562,  0.404813349247,
    -0.481496721506, -0.389882832766,  0.020690204576, -0.772020936012,
    -0.758921504021,  0.323482036591,  0.115715265274, -0.811228036880,
    -0.882436633110,  0.176811277866,  1.678015947342,  0.379081040621,
    -0.842976212502,  0.346952259541, -0.545828759670,  1.632800459862
};

static double GELU_golden_output[] = {
    1.498199582100,  1.014679551125,  0.580462038517, -0.161344811320,
    -0.135453075171,  0.007294139825,  0.225325092673,  1.235459089279,
     0.094946734607,  0.165724009275, -0.133120641112,  1.871103763580,
     0.191376730800, -0.125069886446,  0.169681981206, -0.112644664943,
    -0.154036879539, -0.169163048267,  0.061428427696,  1.132469892502,
    -0.089851818979,  0.552240371704,  0.899579226971, -0.168043658137,
     0.565008401871,  0.048956073821,  0.439583092928, -0.090532489121,
    -0.060955654830,  0.798911273479, -0.064101703465,  0.048816055059,
    -0.156645998359,  0.082529976964, -0.154254898429,  1.232632875443,
    -0.083896033466, -0.123835846782, -0.086161509156,  0.048703473061,
    -0.167972877622, -0.121522113681, -0.046670529991,  0.617986679077,
    -0.157319813967, -0.092503339052,  0.709896743298,  0.574865520000,
     0.703132867813, -0.169963955879,  1.411436080933,  0.580042064190,
     0.741154611111, -0.164741978049,  1.129479527473, -0.069256491959,
    -0.049848672003, -0.169087052345,  0.255214750767, -0.157380074263,
     0.223928079009,  0.632535398006,  0.300378054380, -0.169946283102,
     1.068588852882, -0.165071934462,  0.783203184605,  0.614346146584,
     1.591325283051,  0.219006344676, -0.115003645420,  0.192637458444,
    -0.166712537408, -0.049788996577, -0.169361919165, -0.168130636215,
    -0.155041679740, -0.060888241976, -0.134137839079, -0.160614117980,
    -0.132782235742, -0.156389534473,  0.354075312614,  0.936574816704,
     2.654553413391, -0.092845752835, -0.035900454968, -0.154874503613,
     1.204704761505, -0.062572605908,  1.230982899666, -0.039479542524,
     1.401402950287, -0.120890334249,  0.091938301921, -0.103604510427,
     0.228880971670,  0.108285568655,  1.513783097267, -0.167782157660,
     0.688394129276,  0.593158841133, -0.157540664077,  0.418839782476,
     0.048209801316,  1.920528769493,  0.386099845171, -0.141709372401,
     1.069367766380, -0.049809500575, -0.135230198503,  0.574639260769,
    -0.066881760955,  0.596510827541, -0.162873372436, -0.158483341336,
    -0.169686436653, -0.161375194788,  0.720409095287,  2.304597616196,
    -0.065585561097, -0.164551988244, -0.131098195910, -0.148708447814,
    -0.148663327098,  0.089060656726, -0.101548098028,  1.317959904671,
     0.169103100896,  0.024001283571, -0.158595800400, -0.101909510791,
    -0.162240833044, -0.155090972781, -0.118474565446,  0.221488356590,
     0.365645468235,  0.013248858973, -0.151851043105,  1.946992278099,
    -0.101253561676, -0.006014300976, -0.032804865390,  0.260597169399,
    -0.010922161862, -0.145792976022, -0.165743649006, -0.162226170301,
     1.587365984917, -0.168676435947, -0.168497130275,  0.330191940069,
    -0.149622067809, -0.100989677012,  0.432351946831, -0.093922272325,
     1.023946166039,  0.739726305008,  0.025843897834, -0.117827951908,
    -0.130937814713,  1.356489539146, -0.169726014137,  0.729478538036,
    -0.169943705201,  1.207641005516,  1.249209761620, -0.040288090706,
     0.031292784959,  0.777626037598, -0.107090584934, -0.071350336075,
    -0.129670530558,  0.527676224709,  1.161149263382,  1.134579420090,
    -0.162394225597, -0.144757837057,  0.043603736907, -0.166386902332,
    -0.096278958023,  0.895924389362, -0.158969298005,  0.484089732170,
    -0.090857118368, -0.138206124306,  1.115107178688, -0.025622237474,
     0.472724437714,  1.593463659286,  0.355387806892, -0.140493586659,
    -0.169871479273,  0.088687323034,  0.253673940897, -0.164135158062,
    -0.043161027133, -0.136040985584,  0.685087263584,  1.811169505119,
    -0.138226687908,  0.440080583096, -0.102422207594, -0.016713079065,
     0.549075841904, -0.161096408963,  2.155813455582, -0.005001218989,
     0.083037458360, -0.044870752841,  0.505522191525, -0.145202502608,
     0.623111069202, -0.141991063952, -0.154108211398,  1.597298502922,
    -0.061391282827, -0.167753636837, -0.025704355910,  0.182520583272,
     1.957115054131,  0.932696640491,  0.769961357117, -0.099604383111,
     2.153636932373,  0.175279796124, -0.155551761389, -0.089653611183,
    -0.107515335083, -0.168032020330, -0.113423995674, -0.139319628477,
     0.089841812849, -0.073763631284,  0.211594089866, -0.068651281297,
     0.018605981022,  0.690416753292, -0.150658726692,  0.266040354967,
    -0.151710823178, -0.135800719261,  0.010515870526, -0.169883996248,
    -0.169960290194,  0.202769815922,  0.063187584281, -0.169236257672,
    -0.166577890515,  0.100812792778,  1.599699616432,  0.245525524020,
    -0.168275654316,  0.220552831888, -0.159705042839,  1.549110531807
};

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Epilogue_thread_gelu_taylor, device_f32) {

    int const kN = 256;
    int const kV = 4;

    using Element = float;
    using Func = cutlass::epilogue::thread::GELU_taylor<cutlass::Array<Element, kV>>;

    double tolerance = 0.005;
    
    //
    // Construct workspace
    //
    cutlass::HostTensor<Element, cutlass::layout::RowMajor> tensor_Destination({1, kN});
    cutlass::HostTensor<Element, cutlass::layout::RowMajor> tensor_Source({1, kN});

    for (int i = 0; i < kN; ++i) {
        tensor_Source.host_data(i) = Element(GELU_golden_input[i]);
    }

    tensor_Destination.sync_device();
    tensor_Source.sync_device();

    //
    // Launch the kernel
    //
    dim3 grid(1,1,1);
    dim3 block(kN / kV, 1, 1);

    test_Epilogue_thread_activation<Element, kV, Func><<< grid, block >>>(
        tensor_Destination.device_data(),
        tensor_Source.device_data());

    tensor_Destination.sync_host();

    //
    // Verify
    //

    for (int i = 0; i < kN; ++i) {
        Element input = Element(GELU_golden_input[i]);
        Element got = tensor_Destination.host_data(i);
        Element expected = Element(GELU_golden_output[i]);

        double rel_error = (double(got) - double(expected)) / double(expected);

        double tolerance_override = tolerance;

        switch (i) {
            case 142: tolerance_override = 0.008; break;
            case 203: tolerance_override = 0.03; break;
            case 207: tolerance_override = 0.09; break;
            case 218: tolerance_override = 0.013; break;
        }

        EXPECT_LT(std::abs(rel_error), tolerance_override) 
            << "Input[" << i << "]: " << input << ", Got: " << got << ", expected: " << expected;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Epilogue_thread_gelu_taylor, device_f16) {

    int const kN = 256;
    int const kV = 8;

    using Element = cutlass::half_t;
    using Func = cutlass::epilogue::thread::GELU_taylor<cutlass::Array<Element, kV>>;

    double tolerance = 0.005;

    //
    // Construct workspace
    //
    cutlass::HostTensor<Element, cutlass::layout::RowMajor> tensor_Destination({1, kN});
    cutlass::HostTensor<Element, cutlass::layout::RowMajor> tensor_Source({1, kN});

    for (int i = 0; i < kN; ++i) {
        tensor_Source.host_data(i) = Element(GELU_golden_input[i]);
    }

    tensor_Destination.sync_device();
    tensor_Source.sync_device();

    //
    // Launch the kernel
    //
    dim3 grid(1,1,1);
    dim3 block(kN / kV, 1, 1);

    test_Epilogue_thread_activation<Element, kV, Func><<< grid, block >>>(
        tensor_Destination.device_data(),
        tensor_Source.device_data());

    tensor_Destination.sync_host();

    //
    // Verify
    //

    for (int i = 0; i < kN; ++i) {
        Element input = Element(GELU_golden_input[i]);
        Element got = tensor_Destination.host_data(i);
        Element expected = Element(GELU_golden_output[i]);

        double rel_error = (double(got) - double(expected)) / double(expected);
        
        double tolerance_override = tolerance;

        switch (i) {
            case 36: tolerance_override = 0.006; break;
            case 77: tolerance_override = 0.009; break;
            case 95: tolerance_override = 0.008; break;
            case 112: tolerance_override = 0.007; break;
            case 171: tolerance_override = 0.006; break;
            case 203: tolerance_override = 0.03; break;
            case 207: tolerance_override = 0.15; break;
        }

        EXPECT_LT(std::abs(rel_error), tolerance_override) 
            << "Input[" << i << "]: " << input << ", Got: " << got << ", expected: " << expected;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

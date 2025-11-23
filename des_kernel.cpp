#include "kernel_operator.h"
#include "des_kernel_impl.h"


extern "C" __global__ __aicore__ void des_custom_kernel(GM_ADDR input_routing_weigths, GM_ADDR input_experts_ids,
                                                        GM_ADDR output_routing_weigths, GM_ADDR output_experts_ids,
                                                        GM_ADDR output_experts_num, GM_ADDR workspace, GM_ADDR tiling) {

    GET_TILING_DATA(tiling_data, tiling);
    AscendC::printf("/////////////////////////////////////////////////////////////////////////////////\n");
    AscendC::printf("Call AscendC kernel\n");
    AscendC::printf("/////////////////////////////////////////////////////////////////////////////////\n");
    DesImpl op;
    op.Init(input_routing_weigths, input_experts_ids, output_routing_weigths, output_experts_ids, output_experts_num, tiling_data);
    op.Process();
}

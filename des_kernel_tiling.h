
#include "register/tilingdata_base.h"

namespace optiling {

constexpr uint32_t TENSOR_MAX_DIM_NUM = 2;

BEGIN_TILING_DATA_DEF(DesCustomKernelTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, numRow);
  TILING_DATA_FIELD_DEF(uint32_t, collNum);
  TILING_DATA_FIELD_DEF(uint32_t, oneCoreRowNum);
  TILING_DATA_FIELD_DEF(uint32_t, numCore);
  TILING_DATA_FIELD_DEF(uint32_t, topK);
  TILING_DATA_FIELD_DEF(float, treshold);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, TENSOR_MAX_DIM_NUM, xStrides);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DesCustomKernel, DesCustomKernelTilingData)
}

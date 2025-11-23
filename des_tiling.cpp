
#include "des_custom_kernel_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {

#ifndef UINT_MAX
#define UINT_MAX  (__INT_MAX__  *2U +1U)
#endif

template <typename T> T CeilDiv(T x, T y) { return y == 0 ? x : (x + y - 1) / y; }


    void PrintDesTiling(DesCustomKernelTilingData &tilingData)
    {
        std::cout << "Current DesCustomKernelTilingData:" << std::endl
                  << " numRow = " << tilingData.get_numRow() << std::endl
                  << " collNum = " <<  tilingData.get_collNum() << std::endl
                  << " oneCoreRowNum = " << tilingData.get_oneCoreRowNum() << std::endl
                  << " numCore = " << tilingData.get_numCore() << std::endl
                  << " topK = " << tilingData.get_topK() << std::endl
                  << " treshold = " << tilingData.get_treshold() << std::endl;
    }

    uint32_t CalculateTiling(gert::TilingContext* context, DesCustomKernelTilingData &tilingData)
    {
        // set one core for prototype
        uint32_t coreNum = 1;
        // get input matrix shape
        //////////////////////////////////////////////////////////////////////////////
        auto shapeWeigthMatrix = context->GetInputShape(0)->GetStorageShape();
        auto weigthDimNum = shapeWeigthMatrix.GetDimNum();

        uint32_t currentNumRow = 1;
        for (int i = 0; i < weigthDimNum - 1; i++)
        {
            if ((shapeWeigthMatrix.GetDim(i) < 0) && (currentNumRow > UINT_MAX / shapeWeigthMatrix.GetDim(i)))
                return ge::GRAPH_FAILED;

            currentNumRow *= shapeWeigthMatrix.GetDim(i);
        }
        tilingData.set_numRow(currentNumRow);

        if ((shapeWeigthMatrix.GetDim(weigthDimNum - 1) < 0) && (shapeWeigthMatrix.GetDim(weigthDimNum - 1) > UINT_MAX))
            return ge::GRAPH_FAILED;

        uint32_t currentNumCols = static_cast<uint32_t>(shapeWeigthMatrix.GetDim(weigthDimNum - 1));
        uint32_t currentNumCore = CeilDiv(currentNumRow, CeilDiv(currentNumRow, coreNum));
        uint32_t currentOneCoreRowNum = CeilDiv(currentNumRow, coreNum);

        auto attrs = context->GetAttrs();
        const float* currentTreshold = attrs->GetAttrPointer<float>(0);

        // set calculated param to tiling
        tilingData.set_collNum(currentNumCols);
        tilingData.set_oneCoreRowNum(currentOneCoreRowNum);
        tilingData.set_numCore(currentNumCore);
        tilingData.set_topK(8);
        tilingData.set_treshold(*currentTreshold);
        //////////////////////////////////////////////////////////////////////////////
        // set core nums
        context->SetBlockDim(currentNumCore);
        PrintDesTiling(tilingData);
        //////////////////////////////////////////////////////////////////////////////
        return ge::GRAPH_SUCCESS;
    }



    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        DesCustomKernelTilingData tiling;
        printf("////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////\n");
        printf("Call DES tiling function\n");
        printf("////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////\n");
        if (CalculateTiling(context, tiling) != ge::GRAPH_SUCCESS){
            return ge::GRAPH_FAILED;
        }
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class DesCustomKernel : public OpDef {
public:
    explicit DesCustomKernel(const char* name) : OpDef(name)
    {
        this->Input("input_routing_weigths")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("input_experts_ids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("output_routing_weigths")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("output_experts_ids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("output_experts_num")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("input_treshold").AttrType(OPTIONAL).Float(0.8);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(DesCustomKernel);
}

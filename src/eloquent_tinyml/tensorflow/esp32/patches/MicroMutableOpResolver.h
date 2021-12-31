#ifndef ELOQUENTTINYML_MICROMUTABLEOPRESOLVER_H
#define ELOQUENTTINYML_MICROMUTABLEOPRESOLVER_H


using namespace tflite;
using namespace tflite::ops;
using namespace tflite::ops::micro;


namespace Eloquent {
    namespace TinyML {
        namespace TensorFlow {
            /**
             * Make tflite::MicroMutableOpResolver compatible across the library
             */
            class MicroMutableOpResolver : public tflite::MicroMutableOpResolver {
            public:

                int AddDepthwiseConv2D() {
                    AddBuiltin(BuiltinOperator_DEPTHWISE_CONV_2D, Register_DEPTHWISE_CONV_2D());

                    return 0;
                }

                int AddConv2D() {
                    AddBuiltin(BuiltinOperator_CONV_2D, Register_CONV_2D());

                    return 0;
                }

                int AddAveragePool2D() {
                    AddBuiltin(BuiltinOperator_AVERAGE_POOL_2D, Register_AVERAGE_POOL_2D());

                    return 0;
                }

                int AddFullyConnected() {
                    AddBuiltin(BuiltinOperator_FULLY_CONNECTED, Register_FULLY_CONNECTED(), 1, 4);

                    return 0;
                }

                int AddMaxPool2D() {
                    AddBuiltin(BuiltinOperator_MAX_POOL_2D, Register_MAX_POOL_2D());

                    return 0;
                }

                int AddSoftmax() {
                    AddBuiltin(BuiltinOperator_SOFTMAX, Register_SOFTMAX());

                    return 0;
                }

                int AddLogistic() {
                    AddBuiltin(BuiltinOperator_LOGISTIC, Register_LOGISTIC());

                    return 0;
                }

                int AddSVDF() {
                    AddBuiltin(BuiltinOperator_SVDF, Register_SVDF());

                    return 0;
                }

                int AddAbs() {
                    AddBuiltin(BuiltinOperator_ABS, Register_ABS());

                    return 0;
                }

                int AddSin() {
                    AddBuiltin(BuiltinOperator_SIN, Register_SIN());

                    return 0;
                }

                int AddCos() {
                    AddBuiltin(BuiltinOperator_COS, Register_COS());

                    return 0;
                }

                int AddLog() {
                    AddBuiltin(BuiltinOperator_LOG, Register_LOG());

                    return 0;
                }

                int AddSqrt() {
                    AddBuiltin(BuiltinOperator_SQRT, Register_SQRT());

                    return 0;
                }

                int AddRsqrt() {
                    AddBuiltin(BuiltinOperator_RSQRT, Register_RSQRT());

                    return 0;
                }

                int AddSquare() {
                    AddBuiltin(BuiltinOperator_SQUARE, Register_SQUARE());

                    return 0;
                }

                int AddPRelu() {
                    AddBuiltin(BuiltinOperator_PRELU, Register_PRELU());

                    return 0;
                }

                int AddFloor() {
                    AddBuiltin(BuiltinOperator_FLOOR, Register_FLOOR());

                    return 0;
                }

                int AddMaximum() {
                    AddBuiltin(BuiltinOperator_MAXIMUM, Register_MAXIMUM());

                    return 0;
                }

                int AddMinimum() {
                    AddBuiltin(BuiltinOperator_MINIMUM, Register_MINIMUM());

                    return 0;
                }

                int AddArgMax() {
                    AddBuiltin(BuiltinOperator_ARG_MAX, Register_ARG_MAX());

                    return 0;
                }

                int AddArgMin() {
                    AddBuiltin(BuiltinOperator_ARG_MIN, Register_ARG_MIN());

                    return 0;
                }

                int AddLogicalOr() {
                    AddBuiltin(BuiltinOperator_LOGICAL_OR, Register_LOGICAL_OR());

                    return 0;
                }

                int AddLogicalAnd() {
                    AddBuiltin(BuiltinOperator_LOGICAL_AND, Register_LOGICAL_AND());

                    return 0;
                }

                int AddLogicalNot() {
                    AddBuiltin(BuiltinOperator_LOGICAL_NOT, Register_LOGICAL_NOT());

                    return 0;
                }

                int AddReshape() {
                    AddBuiltin(BuiltinOperator_RESHAPE, Register_RESHAPE());

                    return 0;
                }

                int AddEqual() {
                    AddBuiltin(BuiltinOperator_EQUAL, Register_EQUAL());

                    return 0;
                }

                int AddNotEqual() {
                    AddBuiltin(BuiltinOperator_NOT_EQUAL, Register_NOT_EQUAL());

                    return 0;
                }

                int AddGreater() {
                    AddBuiltin(BuiltinOperator_GREATER, Register_GREATER());

                    return 0;
                }

                int AddGreaterEqual() {
                    AddBuiltin(BuiltinOperator_GREATER_EQUAL, Register_GREATER_EQUAL());

                    return 0;
                }

                int AddLess() {
                    AddBuiltin(BuiltinOperator_LESS, Register_LESS());

                    return 0;
                }

                int AddLessEqual() {
                    AddBuiltin(BuiltinOperator_LESS_EQUAL, Register_LESS_EQUAL());

                    return 0;
                }

                int AddCeil() {
                    AddBuiltin(BuiltinOperator_CEIL, Register_CEIL());

                    return 0;
                }

                int AddRound() {
                    AddBuiltin(BuiltinOperator_ROUND, Register_ROUND());

                    return 0;
                }

                int AddStridedSlice() {
                    AddBuiltin(BuiltinOperator_STRIDED_SLICE, Register_STRIDED_SLICE());

                    return 0;
                }

                int AddPack() {
                    AddBuiltin(BuiltinOperator_PACK, Register_PACK());

                    return 0;
                }

                int AddSplit() {
                    AddBuiltin(BuiltinOperator_SPLIT, Register_SPLIT(), 1, 3);

                    return 0;
                }

                int AddUnpack() {
                    AddBuiltin(BuiltinOperator_UNPACK, Register_UNPACK());

                    return 0;
                }

                int AddNeg() {
                    AddBuiltin(BuiltinOperator_NEG, Register_NEG());

                    return 0;
                }

                int AddAdd() {
                    AddBuiltin(BuiltinOperator_ADD, Register_ADD());

                    return 0;
                }

                int AddQuantize() {
                    AddBuiltin(BuiltinOperator_QUANTIZE, Register_QUANTIZE(), 1, 4);

                    return 0;
                }

                int AddDequantize() {
                    AddBuiltin(BuiltinOperator_DEQUANTIZE, Register_DEQUANTIZE(), 1, 4);

                    return 0;
                }

                int AddRelu() {
                    AddBuiltin(BuiltinOperator_RELU, Register_RELU());

                    return 0;
                }

                int AddRelu6() {
                    AddBuiltin(BuiltinOperator_RELU6, Register_RELU6());

                    return 0;
                }
            };
        }
    }
}

#endif //ELOQUENTTINYML_MICROMUTABLEOPRESOLVER_H

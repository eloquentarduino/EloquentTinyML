#ifndef ELOQUENTTINYML_MUTABLETENSORFLOW_H
#define ELOQUENTTINYML_MUTABLETENSORFLOW_H


namespace Eloquent {
    namespace TinyML {
        namespace TensorFlow {

            /**
             * Run TensorFlow Lite models with MicroMutableOpResolver
             */
            template<size_t numInputs, size_t numOutputs, size_t tensorArenaSize>
            class MutableTensorFlow
                    : public AbstractTensorFlow<MicroMutableOpResolver, numInputs, numOutputs, tensorArenaSize> {

            public:
                int AddAbs() {
                    return this->opResolver.AddAbs();
                }

                int AddAdd() {
                    return this->opResolver.AddAdd();
                }

                int AddArgMax() {
                    return this->opResolver.AddArgMax();
                }

                int AddArgMin() {
                    return this->opResolver.AddArgMin();
                }

                int AddAveragePool2D() {
                    return this->opResolver.AddAveragePool2D();
                }

                int AddCeil() {
                    return this->opResolver.AddCeil();
                }

                int AddCircularBuffer() {
                    return this->opResolver.AddCircularBuffer();
                }

                int AddConcatenation() {
                    return this->opResolver.AddConcatenation();
                }

                int AddConv2D() {
                    return this->opResolver.AddConv2D();
                }

                int AddCos() {
                    return this->opResolver.AddCos();
                }

                int AddDepthwiseConv2D() {
                    return this->opResolver.AddDepthwiseConv2D();
                }

                int AddDequantize() {
                    return this->opResolver.AddDequantize();
                }

                int AddDetectionPostprocess() {
                    return this->opResolver.AddDetectionPostprocess();
                }

                int AddEqual() {
                    return this->opResolver.AddEqual();
                }

                int AddFloor() {
                    return this->opResolver.AddFloor();
                }

                //int AddFullyConnected(const TfLiteRegistration &registration = Register_FULLY_CONNECTED()) {
                //    return this->opResolver.AddFullyConnected(registration);
                //}

                int AddGreater() {
                    return this->opResolver.AddGreater();
                }

                int AddGreaterEqual() {
                    return this->opResolver.AddGreaterEqual();
                }

                int AddHardSwish() {
                    return this->opResolver.AddHardSwish();
                }

                int AddL2Normalization() {
                    return this->opResolver.AddL2Normalization();
                }

                int AddLess() {
                    return this->opResolver.AddLess();
                }

                int AddLessEqual() {
                    return this->opResolver.AddLessEqual();
                }

                int AddLog() {
                    return this->opResolver.AddLog();
                }

                int AddLogicalAnd() {
                    return this->opResolver.AddLogicalAnd();
                }

                int AddLogicalNot() {
                    return this->opResolver.AddLogicalNot();
                }

                int AddLogicalOr() {
                    return this->opResolver.AddLogicalOr();
                }

                int AddLogistic() {
                    return this->opResolver.AddLogistic();
                }

                int AddMaximum() {
                    return this->opResolver.AddMaximum();
                }

                int AddMaxPool2D() {
                    return this->opResolver.AddMaxPool2D();
                }

                int AddMean() {
                    return this->opResolver.AddMean();
                }

                int AddMinimum() {
                    return this->opResolver.AddMinimum();
                }

                int AddMul() {
                    return this->opResolver.AddMul();
                }

                int AddNeg() {
                    return this->opResolver.AddNeg();
                }

                int AddNotEqual() {
                    return this->opResolver.AddNotEqual();
                }

                int AddPack() {
                    return this->opResolver.AddPack();
                }

                int AddPad() {
                    return this->opResolver.AddPad();
                }

                int AddPadV2() {
                    return this->opResolver.AddPadV2();
                }

                int AddPrelu() {
                    return this->opResolver.AddPrelu();
                }

                int AddQuantize() {
                    return this->opResolver.AddQuantize();
                }

                int AddReduceMax() {
                    return this->opResolver.AddReduceMax();
                }

                int AddRelu() {
                    return this->opResolver.AddRelu();
                }

                int AddRelu6() {
                    return this->opResolver.AddRelu6();
                }

                int AddReshape() {
                    return this->opResolver.AddReshape();
                }

                int AddResizeNearestNeighbor() {
                    return this->opResolver.AddResizeNearestNeighbor();
                }

                int AddRound() {
                    return this->opResolver.AddRound();
                }

                int AddRsqrt() {
                    return this->opResolver.AddRsqrt();
                }

                int AddShape() {
                    return this->opResolver.AddShape();
                }

                int AddSin() {
                    return this->opResolver.AddSin();
                }

                int AddSoftmax() {
                    return this->opResolver.AddSoftmax();
                }

                int AddSplit() {
                    return this->opResolver.AddSplit();
                }

                int AddSplitV() {
                    return this->opResolver.AddSplitV();
                }

                int AddSqrt() {
                    return this->opResolver.AddSqrt();
                }

                int AddSquare() {
                    return this->opResolver.AddSquare();
                }

                int AddStridedSlice() {
                    return this->opResolver.AddStridedSlice();
                }

                int AddSub() {
                    return this->opResolver.AddSub();
                }

                int AddSvdf() {
                    return this->opResolver.AddSvdf();
                }

                int AddTanh() {
                    return this->opResolver.AddTanh();
                }

                int AddUnpack() {
                    return this->opResolver.AddUnpack();
                }
            };
        }
    }
}

#endif //ELOQUENTTINYML_MUTABLETENSORFLOW_H

#ifndef ELOQUENTTINYML_TF_H
#define ELOQUENTTINYML_TF_H

#ifndef ELOQUENT_TFLM
#error "You must include either <tflm_esp32.h> or <tflm_cortexm.h>"
#else

#include "./exception.h"
#include "./benchmark.h"

using Eloquent::Error::Exception;
using Eloquent::Extra::Time::Benchmark;
using eloq::tf::newInterpreter;


namespace Eloquent {
    namespace TF {
        /**
         * Run TensorFlow model
         */
        template<uint8_t numOps, size_t tensorArenaSize>
        class Sequential {
        public:
            const Model *model;
            MicroMutableOpResolver<numOps> resolver;
            MicroInterpreter *interpreter;
            TfLiteTensor *in;
            TfLiteTensor *out;
            Exception exception;
            uint8_t arena[tensorArenaSize];
            uint16_t numInputs;
            uint16_t numOutputs;
            uint8_t classification;
            Benchmark benchmark;
            float *outputs;

            /**
             * Constructor
             */
            Sequential() :
                exception("TF"),
                model(nullptr),
                interpreter(nullptr),
                in(nullptr),
                out(nullptr),
                numInputs(0),
                numOutputs(0),
                classification(255),
                outputs(NULL)
            {

            }

            /**
             * Set number of inputs
             */
            void setNumInputs(uint16_t n) {
                numInputs = n;
            }

            /**
             * Set number of outputs
             */
            void setNumOutputs(uint16_t n) {
                numOutputs = n;
            }

            /**
             * Get i-th output
             */
            float output(uint16_t i = 0) {
                if (outputs == NULL || i >= numOutputs)
                    return sqrt(-1);

                return outputs[i];
            }

            /**
             * Init model
             */
            Exception& begin(const unsigned char *data) {
                #ifdef TF_NUM_INPUTS
                if (!numInputs)
                    numInputs = TF_NUM_INPUTS;
                #endif

                #ifdef TF_NUM_OUTPUTS
                if (!numOutputs)
                    numOutputs = TF_NUM_OUTPUTS;
                #endif

                if (!numInputs)
                    return exception.set("You must set the number of inputs");

                if (!numOutputs)
                    return exception.set("You must set the number of outputs");

                #ifdef TF_OP_ADD
                resolver.AddAdd();
                #endif
                #ifdef TF_OP_AVERAGEPOOL2D
                resolver.AddAveragePool2D();
                #endif
                #ifdef TF_OP_CONCATENATION
                resolver.AddConcatenation();
                #endif
                #ifdef TF_OP_CONV2D
                resolver.AddConv2D();
                #endif
                #ifdef TF_OP_DEPTHWISECONV2D
                resolver.AddDepthwiseConv2D();
                #endif
                #ifdef TF_OP_ELU
                resolver.AddElu();
                #endif
                #ifdef TF_OP_FULLYCONNECTED
                resolver.AddFullyConnected();
                #endif
                #ifdef TF_OP_LEAKYRELU
                resolver.AddLeakyRelu();
                #endif
                #ifdef TF_OP_MAXPOOL2D
                resolver.AddMaxPool2D();
                #endif
                #ifdef TF_OP_MAXIMUM
                resolver.AddMaximum();
                #endif
                #ifdef TF_OP_MINIMUM
                resolver.AddMinimum();
                #endif
                #ifdef TF_OP_RELU
                resolver.AddRelu();
                #endif
                #ifdef TF_OP_RESHAPE
                resolver.AddReshape();
                #endif
                #ifdef TF_OP_SOFTMAX
                resolver.AddSoftmax();
                #endif

                model = tflite::GetModel(data);

                if (model->version() != TFLITE_SCHEMA_VERSION)
                    return exception.set(String("Model version mismatch. Expected ") + TFLITE_SCHEMA_VERSION + ", got " + model->version());

                interpreter = newInterpreter<numOps>(&resolver, model, arena, tensorArenaSize);

                if (interpreter->AllocateTensors() != kTfLiteOk)
                    return exception.set("AllocateTensors() failed");

                in = interpreter->input(0);
                out = interpreter->output(0);

                // allocate outputs
                outputs = (float*) calloc(numOutputs, sizeof(float));

                return exception.clear();
            }

            /**
             *
             */
            Exception& predict(float *x) {
                for (uint16_t i = 0; i < numInputs; i++)
                    in->data.f[i] = x[i];

                benchmark.start();

                if (interpreter->Invoke() != kTfLiteOk)
                    return exception.set("Invoke() failed");

                for (uint16_t i = 0; i < numOutputs; i++) {
                    outputs[i] = out->data.f[i];
                }

                getClassificationResult();
                benchmark.stop();

                return exception.clear();
            }

            /**
             *
             */
            Exception& predict(int8_t *x) {
                // quantization
                const float inputScale = in->params.scale;
                const float inputOffset = in->params.zero_point;
                const float outputScale = out->params.scale;
                const float outputOffset = out->params.zero_point;

                memcpy(in->data.int8, x, sizeof(int8_t) * numInputs);

                // execute
                benchmark.start();

                if (interpreter->Invoke() != kTfLiteOk)
                    return exception.set("Invoke() failed");

                for (uint16_t i = 0; i < numOutputs; i++) {
                    outputs[i] = out->data.int8[i];
                }

                getClassificationResult();
                benchmark.stop();

                return exception.clear();
            }

        protected:

            /**
             * If classification task, get most probable class
             */
            void getClassificationResult() {
                if (numOutputs < 2)
                    return;

                float maxProba = outputs[0];
                classification = 0;

                for (uint16_t i = 1; i < numOutputs; i++) {
                    if (outputs[i] > maxProba) {
                        classification = i;
                        maxProba = outputs[i];
                    }
                }
            }
        };
    } // namespace TF
} // namespace Eloquent

#endif // ELOQUENT_TFLM
#endif //ELOQUENTTINYML_TF_H

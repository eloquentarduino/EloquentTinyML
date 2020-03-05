#pragma once

#include <Arduino.h>
#include <math.h>
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"


namespace Eloquent {
    namespace TinyML {

        /**
         * Eloquent interface to Tensorflow Lite for Microcontrollers
         *
         * @tparam inputSize
         * @tparam outputSize
         * @tparam tensorArenaSize how much memory to allocate to the tensors
         */
        template<size_t inputSize, size_t outputSize, size_t tensorArenaSize>
        class TfLite {
        public:
            /**
             * Contructor
             * @param modelData a model as exported by tinymlgen
             */
            TfLite(unsigned char *modelData) {
                static tflite::MicroErrorReporter microReporter;
                static tflite::ops::micro::AllOpsResolver resolver;

                reporter = &microReporter;
                model = tflite::GetModel(modelData);

                // assert model version and runtime version match
                if (model->version() != TFLITE_SCHEMA_VERSION) {
                  failed = true;
                  reporter->Report(
                      "Model provided is schema version %d not equal "
                      "to supported version %d.",
                      model->version(), TFLITE_SCHEMA_VERSION);

                  return;
                }

                static tflite::MicroInterpreter interpreter(model, resolver, tensorArena, tensorArenaSize, reporter);

                if (interpreter.AllocateTensors() != kTfLiteOk) {
                    failed = true;
                    reporter->Report("AllocateTensors() failed");
                    return;
                }

                input = interpreter.input(0);
                output = interpreter.output(0);
                this->interpreter = &interpreter;
            }

            /**
             * Test if the initialization completed fine
             */
            bool initialized() {
                return !failed;
            }

            /**
             * Run inference
             * @return output[0], so you can use it directly if it's the only output
             */
            float predict(float *input, float *output = NULL) {
                // abort if initialization failed
                if (failed)
                    return sqrt(-1);

                // copy input
                for (size_t i = 0; i < inputSize; i++)
                    this->input->data.f[i] = input[i];

                if (interpreter->Invoke() != kTfLiteOk) {
                    reporter->Report("Inference failed");

                    return sqrt(-1);
                }

                // copy output
                if (output != NULL) {
                    for (size_t i = 0; i < outputSize; i++)
                        output[i] = this->output->data.f[i];
                }

                return this->output->data.f[0];
            }

        protected:
            bool failed;
            uint8_t tensorArena[tensorArenaSize];
            tflite::ErrorReporter *reporter;
            tflite::MicroInterpreter *interpreter;
            TfLiteTensor *input;
            TfLiteTensor *output;
            const tflite::Model *model;
        };
    }
}

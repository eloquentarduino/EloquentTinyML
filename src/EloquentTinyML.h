#pragma once

#include <Arduino.h>
#include <math.h>
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"


namespace Eloquent {
    namespace TinyML {

        enum TfLiteError {
            OK,
            VERSION_MISMATCH,
            CANNOT_ALLOCATE_TENSORS,
            NOT_INITIALIZED,
            INVOKE_ERROR
        };

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
            TfLite() :
                failed(false) {
            }

            /**
             * Inizialize NN
             *
             * @param modelData
             * @return
             */
            bool begin(const unsigned char *modelData) {
                static tflite::MicroErrorReporter microReporter;
                static tflite::ops::micro::AllOpsResolver resolver;

                reporter = &microReporter;
                model = tflite::GetModel(modelData);

                // assert model version and runtime version match
                if (model->version() != TFLITE_SCHEMA_VERSION) {
                    failed = true;
                    error = VERSION_MISMATCH;

                    reporter->Report(
                            "Model provided is schema version %d not equal "
                            "to supported version %d.",
                            model->version(), TFLITE_SCHEMA_VERSION);

                    return false;
                }

                static tflite::MicroInterpreter interpreter(model, resolver, tensorArena, tensorArenaSize, reporter);

                if (interpreter.AllocateTensors() != kTfLiteOk) {
                    failed = true;
                    error = CANNOT_ALLOCATE_TENSORS;

                    return false;
                }

                input = interpreter.input(0);
                output = interpreter.output(0);
                error = OK;

                this->interpreter = &interpreter;

                return true;
            }

            /**
             * Test if the initialization completed fine
             */
            bool initialized() {
                return !failed;
            }

            /**
             *
             * @param input
             * @param output
             * @return
             */
            uint8_t predict(uint8_t *input, uint8_t *output = NULL) {
                // abort if initialization failed
                if (!initialized())
                    return sqrt(-1);

                memcpy(this->input->data.uint8, input, sizeof(uint8_t) * inputSize);

                if (interpreter->Invoke() != kTfLiteOk) {
                    reporter->Report("Inference failed");

                    return sqrt(-1);
                }

                // copy output
                if (output != NULL) {
                    for (uint16_t i = 0; i < outputSize; i++)
                        output[i] = this->output->data.uint8[i];
                }

                return this->output->data.uint8[0];
            }

            /**
             * Run inference
             * @return output[0], so you can use it directly if it's the only output
             */
            float predict(float *input, float *output = NULL) {
                // abort if initialization failed
                if (!initialized()) {
                    error = NOT_INITIALIZED;

                    return sqrt(-1);
                }

                // copy input
                for (size_t i = 0; i < inputSize; i++)
                    this->input->data.f[i] = input[i];

                if (interpreter->Invoke() != kTfLiteOk) {
                    error = INVOKE_ERROR;
                    reporter->Report("Inference failed");

                    return sqrt(-1);
                }

                // copy output
                if (output != NULL) {
                    for (uint16_t i = 0; i < outputSize; i++)
                        output[i] = this->output->data.f[i];
                }

                return this->output->data.f[0];
            }

            /**
             * Predict class
             * @param input
             * @return
             */
            uint8_t predictClass(float *input) {
                float output[outputSize];

                predict(input, output);

                return probaToClass(output);
            }

            /**
             * Get class with highest probability
             * @param output
             * @return
             */
            uint8_t probaToClass(float *output) {
                uint8_t classIdx = 0;
                float maxProba = output[0];

                for (uint8_t i = 1; i < outputSize; i++) {
                    if (output[i] > maxProba) {
                        classIdx = i;
                        maxProba = output[i];
                    }
                }

                return classIdx;
            }

            /**
             * Get error message
             * @return
             */
            const char* errorMessage() {
                switch (error) {
                    case OK:
                        return "No error";
                    case VERSION_MISMATCH:
                        return "Version mismatch";
                    case CANNOT_ALLOCATE_TENSORS:
                        return "Cannot allocate tensors";
                    case NOT_INITIALIZED:
                        return "Interpreter has not been initialized";
                    case INVOKE_ERROR:
                        return "Interpreter invoke() returned an error";
                    default:
                        return "Unknown error";
                }
            }

        protected:
            bool failed;
            TfLiteError error;
            uint8_t tensorArena[tensorArenaSize];
            tflite::ErrorReporter *reporter;
            tflite::MicroInterpreter *interpreter;
            TfLiteTensor *input;
            TfLiteTensor *output;
            const tflite::Model *model;
        };
    }
}

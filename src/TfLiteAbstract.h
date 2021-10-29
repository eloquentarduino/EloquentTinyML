//
// Created by Simone on 28/10/2021.
//

#ifndef ELOQUENTTINYML_TFLITEABSTRACT_H
#define ELOQUENTTINYML_TFLITEABSTRACT_H


namespace Eloquent {
    namespace TinyML {

        /**
         *
         */
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
        template<class OpsResolver, size_t inputSize, size_t outputSize, size_t tensorArenaSize>
        class TfLiteAbstract {
        public:
            /**
             * Contructor
             */
            TfLiteAbstract() :
                failed(false),
                shouldRescaleInput(false),
                shouldRescaleOutput(false) {
            }

            /**
             * Destructor
             */
            ~TfLiteAbstract() {
                delete interpreter;
                delete model;
            }

            /**
             * Inizialize NN
             *
             * @param modelData
             * @return
             */
            bool begin(const unsigned char *modelData) {
                model = tflite::GetModel(modelData);

                if (model->version() != TFLITE_SCHEMA_VERSION)
                    return this->abort(VERSION_MISMATCH, false);

                interpreter = new tflite::MicroInterpreter(model, opsResolver, tensorArena, tensorArenaSize, &errorReporter);

                if (interpreter->AllocateTensors() != kTfLiteOk)
                    return this->abort(CANNOT_ALLOCATE_TENSORS, false);

                input = interpreter->input(0);
                output = interpreter->output(0);
                error = OK;
                failed = false;

                return true;
            }

            /**
             * Test if the initialization completed fine
             */
            bool isInitialized() {
                return !failed;
            }

            /**
             *
             * @param on
             */
            void turnInputScalingOn(bool on = true) {
                shouldRescaleInput = on;
            }

            /**
             *
             * @param on
             */
            void turnOutputScalingOn(bool on = true) {
                shouldRescaleOutput = on;
            }

            /**
             *
             * @param input
             * @param output
             * @return
             */
            uint8_t predict(uint8_t *input, uint8_t *output = NULL) {
                if (!isInitialized())
                    return this->abort(NOT_INITIALIZED, 255);

                memcpy(this->input->data.uint8, input, sizeof(uint8_t) * inputSize);

                if (interpreter->Invoke() != kTfLiteOk)
                    return this->abort(INVOKE_ERROR, 255);

                // copy output
                if (output != NULL) {
                    for (uint16_t i = 0; i < outputSize; i++) {
                        uint8_t y = this->output->data.uint8[i];

                        output[i] = shouldRescaleOutput ? scaleOutput(y) : y;
                    }
                }

                return this->output->data.uint8[0];
            }

            /**
             *
             * @param input
             * @param output
             * @return
             */
            int8_t predict(int8_t *input, int8_t *output = NULL) {
                if (!isInitialized())
                    return this->abort(NOT_INITIALIZED, -127);

                memcpy(this->input->data.int8, input, sizeof(int8_t) * inputSize);

                if (interpreter->Invoke() != kTfLiteOk)
                    return this->abort(INVOKE_ERROR, -127);

                // copy output
                if (output != NULL) {
                    for (uint16_t i = 0; i < outputSize; i++) {
                        int8_t y = this->output->data.int8[i];

                        output[i] = shouldRescaleOutput ? scaleOutput(y) : y;
                    }
                }

                return this->output->data.int8[0];
            }

            /**
             * Run inference
             * @return output[0], so you can use it directly if it's the only output
             */
            float predict(float *input, float *output = NULL) {
                if (!isInitialized())
                    return this->abort(NOT_INITIALIZED, sqrt(-1));

                // copy input
                for (size_t i = 0; i < inputSize; i++)
                    this->input->data.f[i] = input[i];

                if (interpreter->Invoke() != kTfLiteOk)
                    return this->abort(INVOKE_ERROR, sqrt(-1));

                // copy output
                if (output != NULL) {
                    for (uint16_t i = 0; i < outputSize; i++) {
                        float y = this->output->data.f[i];

                        output[i] = shouldRescaleOutput ? scaleOutput(y) : y;
                    }
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
             * Apply model scaling to input
             * @tparam T
             * @param x
             * @return
             */
            template<typename T>
            T scaleInput(T x) {
                return x / this->input->params.zero_point + this->input->params.scale;
            }

            /**
             * Apply model scaling to output
             * @tparam T
             * @param y
             * @return
             */
            template<typename T>
            T scaleOutput(T y) {
                return (y - this->output->params.zero_point) * this->output->params.scale;
            }

            /**
             * Get error
             * @return
             */
            TfLiteError getError() {
                return error;
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
            bool shouldRescaleInput;
            bool shouldRescaleOutput;
            TfLiteError error;
            uint8_t tensorArena[tensorArenaSize];
            tflite::MicroErrorReporter errorReporter;
            tflite::MicroInterpreter *interpreter;
            TfLiteTensor *input;
            TfLiteTensor *output;
            const tflite::Model *model;
            OpsResolver opsResolver;

            /**
             * Abort execution with given error code
             *
             * @tparam T
             * @param errorCode
             * @param rvalue
             * @return
             */
            template<typename T>
            T abort(TfLiteError errorCode, T rvalue) {
                error = errorCode;
                failed = true;

                return rvalue;
            }
        };
    }
}

#endif //ELOQUENTTINYML_TFLITE_H

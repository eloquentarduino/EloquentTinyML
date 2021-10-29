//
// Created by Simone on 28/10/2021.
//

#ifndef ELOQUENTTINYML_TFLITEESP32_H
#define ELOQUENTTINYML_TFLITEESP32_H

#include "tensorflow_esp32/tensorflow/lite/version.h"
#include "tensorflow_esp32/tensorflow/lite/schema/schema_generated.h"
#include "tensorflow_esp32/tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow_esp32/tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow_esp32/tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "TfLiteAbstract.h"


namespace Eloquent {
    namespace TinyML {

        /**
         * Run TensorFlow Lite models on ESP32
         */
        template<size_t inputSize, size_t outputSize, size_t tensorArenaSize>
        class TfLite : public TfLiteAbstract<tflite::ops::micro::AllOpsResolver, inputSize, outputSize, tensorArenaSize> {
        };
    }
}

#endif //ELOQUENTTINYML_TFLITEESP32_H

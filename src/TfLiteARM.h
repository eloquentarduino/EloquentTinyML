//
// Created by Simone on 28/10/2021.
//

#ifndef ELOQUENTTINYML_TFLITEARM_H
#define ELOQUENTTINYML_TFLITEARM_H

#include "tensorflow_arm/tensorflow/lite/version.h"
#include "tensorflow_arm/tensorflow/lite/schema/schema_generated.h"
#include "tensorflow_arm/tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow_arm/tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow_arm/tensorflow/lite/micro/micro_interpreter.h"
#include "TfLiteAbstract.h"


namespace Eloquent {
    namespace TinyML {

        /**
         * Run TensorFlow Lite models on ARM
         */
        template<size_t inputSize, size_t outputSize, size_t tensorArenaSize>
        class TfLite : public TfLiteAbstract<tflite::AllOpsResolver, inputSize, outputSize, tensorArenaSize> {
        };
    }
}

#endif //ELOQUENTTINYML_TFLITEESP32_H

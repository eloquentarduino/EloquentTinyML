//
// Created by Simone on 28/10/2021.
//

#ifndef ELOQUENTTINYML_ALLOPSTENSORFLOW_H
#define ELOQUENTTINYML_ALLOPSTENSORFLOW_H


namespace Eloquent {
    namespace TinyML {
        namespace TensorFlow {

            /**
             * Run TensorFlow Lite models with AllOpsResolver
             */
            template<size_t numInputs, size_t numOutputs, size_t tensorArenaSize>
            class AllOpsTensorFlow
                    : public AbstractTensorFlow<AllOpsResolver, numInputs, numOutputs, tensorArenaSize> {
            };


            /**
             * An alias to AllOpsResolver
             */
            template<size_t numInputs, size_t numOutputs, size_t tensorArenaSize>
            class TensorFlow
                    : public AbstractTensorFlow<AllOpsResolver, numInputs, numOutputs, tensorArenaSize> {
            };
        }
    }
}

#endif //ELOQUENTTINYML_ALLOPSTENSORFLOW_H

//
// Created by Simone on 09/11/2021.
//

#ifndef ELOQUENTTINYML_ALLOPSRESOLVER_H
#define ELOQUENTTINYML_ALLOPSRESOLVER_H

namespace Eloquent {
    namespace TinyML {
        namespace TensorFlow {
            /**
             * Make tflite::AllOpsResolver compatible across the library
             */
            class AllOpsResolver : public tflite::ops::micro::AllOpsResolver {

            };
        }
    }
}

#endif //ELOQUENTTINYML_ALLOPSRESOLVER_H

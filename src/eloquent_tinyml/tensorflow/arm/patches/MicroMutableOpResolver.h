//
// Created by Simone on 01/11/2021.
//

#ifndef ELOQUENTTINYML_MICROMUTABLEOPRESOLVER_H
#define ELOQUENTTINYML_MICROMUTABLEOPRESOLVER_H


namespace Eloquent {
    namespace TinyML {
        namespace TensorFlow {
            /**
             * Make tflite::MicroMutableOpResolver compatible across the library
             */
            class MicroMutableOpResolver : public tflite::MicroMutableOpResolver<128> {

            };
        }
    }
}

#endif //ELOQUENTTINYML_MICROMUTABLEOPRESOLVER_H

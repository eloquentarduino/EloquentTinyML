#pragma once

#include "EloquentTinyML.h"
#include "persondetection/model.h"

// Keeping these as constant expressions allow us to allocate fixed-sized arrays
// on the stack for our working memory.

// All of these values are derived from the values used during model training,
// if you change your model you'll need to update these constants.
const int kNumCols = 96;
const int kNumRows = 96;
const int kNumChannels = 1;
const int kMaxImageSize = kNumCols * kNumRows * kNumChannels;
const int kCategoryCount = 3;
const int kPersonIndex = 1;
const int kNotAPersonIndex = 2;
const char* kCategoryLabels[kCategoryCount] = {
        "unused",
        "person",
        "notperson",
};


namespace Eloquent {
    namespace TinyML {

        /**
         *
         * @tparam tensorArenaSize
         */
        template<size_t tensorArenaSize>
        class PersonDetector {
        public:
            PersonDetector() {

            }

            /**
             *
             * @return
             */
            bool begin() {
                return tf.begin(g_person_detect_model_data);
            }

            /**
             *
             * @return
             */
            bool initialized() {
                return tf.initialized();
            }

            void detect(uint8_t *grayscaleImage) {
                tf.predict(grayscaleImage, scores);
            }

            /**
             * 
             * @return 
             */
            uint8_t getPersonScore() {
                return scores[1];
            }

            /**
             * 
             * @return 
             */
            uint8_t getNonPersonScore() {
                return scores[2];
            }
            
            /**
             * 
             * @return 
             */
            bool isPerson() {
                return getPersonScore() > getNonPersonScore();
            }

        protected:
            uint8_t scores[3];
            TfLite<kNumRows * kNumCols, 3, tensorArenaSize> tf;
        };
    }
}
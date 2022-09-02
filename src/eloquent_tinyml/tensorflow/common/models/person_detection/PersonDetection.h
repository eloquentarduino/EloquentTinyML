#ifndef ELOQUENTTINYML_TENSORFLOWPERSONDETECTION_H
#define ELOQUENTTINYML_TENSORFLOWPERSONDETECTION_H

#include "./person_detection_model.h"

#ifndef PERSON_DETECTION_ARENA_SIZE
#define PERSON_DETECTION_ARENA_SIZE 90000
#endif


namespace Eloquent {
    namespace TinyML {
        namespace TensorFlow {

            /**
             * Person detection error codes
             */
            enum PersonDetectionError {
                PERSON_DETECTION_OK,
                PERSON_DETECTION_CANNOT_INIT_NETWORK,
                PERSON_DETECTION_IMAGE_SIZE_MISMATCH
            };

            enum PersonDetectionResizeStrategy {
                PERSON_DETECTION_CROP_TO_CENTER,
                PERSON_DETECTION_CROP_UNIFORM
            };

            /**
             * Perform person detection on images
             * @tparam imageWidth
             * @tparam imageHeight
             */
            template<uint16_t imageWidth, uint16_t imageHeight>
            class PersonDetection {
            public:
                PersonDetection() :
                    error(PERSON_DETECTION_OK) {

                }

                /**
                 * Init network
                 * @return true if ok, false otherwise
                 */
                bool begin() {
                    tf.AddDepthwiseConv2D();
                    tf.AddConv2D();
                    tf.AddAveragePool2D();

                    if (!tf.begin(g_person_detect_model_data)) {
                        error = PersonDetectionError::PERSON_DETECTION_CANNOT_INIT_NETWORK;
                    }

                    return isOk();
                }

                /**
                 *
                 * @param threshold
                 */
                void setDetectionAbsoluteThreshold(uint8_t threshold) {
                    absoluteThreshold = threshold;
                }

                /**
                 *
                 * @param threshold
                 */
                void setDetectionDifferenceThreshold(uint8_t threshold) {
                    differenceThreshold = threshold;
                }

                /**
                 *
                 * @param threshold
                 */
                void setDetectionRelativeThreshold(float threshold) {
                    relativeThreshold = threshold;
                }

                /**
                 *
                 */
                void cropToCenter() {
                    resizeStrategy = PersonDetectionResizeStrategy::PERSON_DETECTION_CROP_TO_CENTER;
                }

                /**
                 *
                 */
                void resizeUniform() {
                    resizeStrategy = PersonDetectionResizeStrategy::PERSON_DETECTION_CROP_UNIFORM;
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
                uint8_t getNotPersonScore() {
                    return scores[2];
                }

                /**
                 * Detect if there is a person in the image
                 * @return
                 */
                bool detectPerson(uint8_t *image) {
                    if (imageWidth < 96 || imageHeight < 96) {
                        error = PersonDetectionError::PERSON_DETECTION_IMAGE_SIZE_MISMATCH;
                        return false;
                    }

                    error = PersonDetectionError::PERSON_DETECTION_OK;
                    crop(image);

                    // run inference
                    uint32_t startTime = millis();
                    tf.predict(image, scores);
                    elapsedTime = millis() - startTime;

                    // apply threshold
                    uint8_t person = getPersonScore();
                    uint8_t notPerson = getNotPersonScore();

                    if (notPerson > person)
                        return false;

                    if (absoluteThreshold > 0) {
                        return person >= absoluteThreshold;
                    }

                    if (differenceThreshold > 0) {
                        return person - notPerson >= differenceThreshold;
                    }

                    if (relativeThreshold > 0) {
                        return person >= relativeThreshold * notPerson;
                    }

                    return true;
                }

                /**
                 *
                 * @return
                 */
                uint32_t getElapsedTime() {
                    return elapsedTime;
                }

                /**
                 * Test if an error occurred
                 * @return
                 */
                bool isOk() {
                    return error == PersonDetectionError::PERSON_DETECTION_OK;
                }

                /**
                 *
                 * @return
                 */
                PersonDetectionError getError() {
                    return error;
                }

                /**
                 *
                 * @return
                 */
                const char *getErrorMessage() {
                    switch (error) {
                        case PersonDetectionError::PERSON_DETECTION_OK:
                            return "OK";
                        case PersonDetectionError::PERSON_DETECTION_CANNOT_INIT_NETWORK:
                            return "Cannot init network";
                        case PersonDetectionError::PERSON_DETECTION_IMAGE_SIZE_MISMATCH:
                            return "Input image MUST be at least 96x96";
                        default:
                            return "Unknown error";
                    }
                }

            protected:
                bool inited = false;
                uint8_t scores[3];
                uint32_t elapsedTime = 0;
                uint8_t absoluteThreshold = 0;
                uint8_t differenceThreshold = 0;
                PersonDetectionResizeStrategy resizeStrategy = PersonDetectionResizeStrategy::PERSON_DETECTION_CROP_TO_CENTER;
                float relativeThreshold = 0;
                PersonDetectionError error;
                MutableTensorFlow<96 * 96, 3, PERSON_DETECTION_ARENA_SIZE> tf;

                /**
                 * In-place crop
                 * @param image
                 */
                void crop(uint8_t *image) {
                    if (imageWidth == 96 && imageHeight == 96) {
                        return;
                    }

                    // only use center region of image
                    else if (resizeStrategy == PersonDetectionResizeStrategy::PERSON_DETECTION_CROP_TO_CENTER) {
                        const uint16_t xOffset = (imageWidth - 96) / 2;
                        const uint16_t xEnd = xOffset + 96;
                        const uint16_t yOffset = (imageHeight - 96) / 2;
                        const uint16_t yEnd = yOffset + 96;
                        uint16_t i = 0;

                        for (uint16_t y = yOffset; y < yEnd; y++) {
                            const uint16_t offset = y * imageWidth;

                            for (uint16_t x = xOffset; x < xEnd; x++) {
                                image[i++] = image[offset + x];
                            }
                        }
                    }
                    // use 1 pixel every nth
                    else if (resizeStrategy == PersonDetectionResizeStrategy::PERSON_DETECTION_CROP_UNIFORM) {
                        const float dx = ((float) imageWidth) / 96;
                        const float dy = ((float) imageHeight) / 96;

                        for (int i = 0; i < 96; i++) {
                            const uint16_t y = dy * i;
                            const uint16_t srcOffset = y * imageWidth;
                            const uint16_t destOffset = i * 96;

                            for (int j = 0; j < 96; j++) {
                                const uint16_t x = dx * j;

                                image[destOffset + j] = image[srcOffset + x];
                            }
                        }
                    }
                }
            };
        }
    }
}

#endif //ELOQUENTTINYML_TENSORFLOWPERSONDETECTION_H

#include <EloquentTinyML.h>
#include <eloquent_tinyml/tensorflow/person_detection.h>

#if defined(ESP32)
#include "ESP32Camera.h"
#else
#include "PortentaVision.h"
#endif

const uint16_t imageWidth = 320;
const uint16_t imageHeight = 240;


Eloquent::TinyML::TensorFlow::PersonDetection<imageWidth, imageHeight> detector;


void setup() {
    Serial.begin(115200);
    delay(5000);
    initCamera();

    // configure a threshold for "robust" person detection
    // if no threshold is set, "person" would be detected everytime person_score > not_person_score
    // even if just by 1
    // by trial and error, considering that scores range from 0 to 255, a threshold of 190-200
    // dramatically reduces the number of false positives
    detector.setDetectionAbsoluteThreshold(190);
    detector.begin();

    // abort if an error occurred
    if (!detector.isOk()) {
        Serial.print("Setup error: ");
        Serial.println(detector.getErrorMessage());

        while (true) delay(1000);
    }
}

void loop() {
    uint8_t *frame = captureFrame();
    bool isPersonInFrame = detector.detectPerson(frame);

    if (!detector.isOk()) {
        Serial.print("Loop error: ");
        Serial.println(detector.getErrorMessage());

        delay(10000);
        return;
    }

    Serial.println(isPersonInFrame ? "Person detected" : "No person detected");
    Serial.print("\t > It took ");
    Serial.print(detector.getElapsedTime());
    Serial.println("ms to detect");
    Serial.print("\t > Person score: ");
    Serial.println(detector.getPersonScore());
    Serial.print("\t > Not person score: ");
    Serial.println(detector.getNotPersonScore());
    delay(1000);
}
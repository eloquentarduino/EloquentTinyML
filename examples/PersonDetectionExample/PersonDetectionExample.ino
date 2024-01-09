/**
 * Run person detection on ESP32 camera
 *  - Requires tflm_esp32 library
 *  - Requires EloquentEsp32Cam library
 *
 * Detections takes about 4-5 seconds per frame
 */
#include <Arduino.h>
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>
#include <eloquent_tinyml/zoo/person_detection.h>
#include <eloquent_esp32cam.h>

using eloq::camera;
using eloq::tinyml::zoo::personDetection;


void setup() {
    delay(3000);
    Serial.begin(115200);
    Serial.println("__PERSON DETECTION__");

    // camera settings
    // replace with your own model!
    camera.pinout.freenove_s3();
    camera.brownout.disable();
    // only works on 96x96 (yolo) grayscale images
    camera.resolution.yolo();
    camera.pixformat.gray();

    // init camera
    while (!camera.begin().isOk())
        Serial.println(camera.exception.toString());

    // init tf model
    while (!personDetection.begin().isOk())
        Serial.println(personDetection.exception.toString());

    Serial.println("Camera OK");
    Serial.println("Point the camera to yourself");
}

void loop() {
    Serial.println("Loop");

    // capture picture
    if (!camera.capture().isOk()) {
        Serial.println(camera.exception.toString());
        return;
    }

    // run person detection
    if (!personDetection.run(camera).isOk()) {
        Serial.println(personDetection.exception.toString());
        return;
    }

    // a person has been detected!
    if (personDetection) {
        Serial.print("Person detected in ");
        Serial.print(personDetection.tf.benchmark.millis());
        Serial.println("ms");
    }
}
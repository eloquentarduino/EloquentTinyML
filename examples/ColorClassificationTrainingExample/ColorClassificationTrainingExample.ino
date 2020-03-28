#include <EloquentSVMSMO.h>
#include "RGB.h"

#define MAX_TRAINING_SAMPLES 20
#define FEATURES_DIM 3


int numSamples;
RGB rgb(2, 3, 4);
float X_train[MAX_TRAINING_SAMPLES][FEATURES_DIM];
int y_train[MAX_TRAINING_SAMPLES];
Eloquent::TinyML::SVMSMO<FEATURES_DIM> classifier(linearKernel);


void setup() {
    Serial.begin(115200);
    rgb.begin();

    classifier.setC(5);
    classifier.setTol(1e-5);
    classifier.setMaxIter(10000);
}

void loop() {
    if (!Serial.available()) {
        delay(100);
        return;
    }

    String command = Serial.readStringUntil('\n');

    if (command == "help") {
        Serial.println("Available commands:");
        Serial.println("\tfit: train the classifier on a new set of samples");
        Serial.println("\tpredict: classify a new sample");
        Serial.println("\tinspect: print X_train and y_train");
    }
    else if (command == "fit") {
        Serial.print("How many samples will you record? ");
        numSamples = readSerialNumber();

        Serial.print("You will record ");
        Serial.print(numSamples);
        Serial.println(" samples");

        for (int i = 0; i < numSamples; i++) {
            Serial.println("Which class does the sample belongs to, 1 or -1?");
            y_train[i] = readSerialNumber() > 0 ? 1 : -1;
            getFeatures(X_train[i]);
        }

        Serial.print("Start training... ");
        classifier.fit(X_train, y_train, numSamples);
        Serial.println("Done");
    }
    else if (command == "predict") {
        int label;
        float x[FEATURES_DIM];

        getFeatures(x);
        Serial.print("Predicted label is ");
        Serial.println(classifier.predict(X_train, x));
    }
    else if (command == "inspect") {
        for (int i = 0; i < numSamples; i++) {
            Serial.print("[");
            Serial.print(y_train[i]);
            Serial.print("] ");

            for (int j = 0; j < FEATURES_DIM; j++) {
                Serial.print(X_train[i][j]);
                Serial.print(", ");
            }

            Serial.println();
        }
    }
}

/**
 *
 * @return
 */
int readSerialNumber() {
    while (!Serial.available()) delay(1);

    return Serial.readStringUntil('\n').toInt();
}

/**
 * Get features for new sample
 * @param x
 */
void getFeatures(float x[FEATURES_DIM]) {
    rgb.read(x);

    for (int i = 0; i < FEATURES_DIM; i++) {
        Serial.print(x[i]);
        Serial.print(", ");
    }

    Serial.println();
}
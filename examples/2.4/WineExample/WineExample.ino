#include <EloquentTinyML.h>
#include <eloquent_tinyml/tensorflow.h>

// wine_model.h contains the array you exported from Python with xxd or tinymlgen
#include "wine_model.h"

#define N_INPUTS 13
#define N_OUTPUTS 3
#define TENSOR_ARENA_SIZE 16*1024

Eloquent::TinyML::TensorFlow::TensorFlow<N_INPUTS, N_OUTPUTS, TENSOR_ARENA_SIZE> tf;

float X_test[10][13] = {
        {14.23, 1.71, 2.43, 15.60, 127.00, 2.80, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.00},
        {13.20, 1.78, 2.14, 11.20, 100.00, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.40, 1050.00},
        {13.16, 2.36, 2.67, 18.60, 101.00, 2.80, 3.24, 0.30, 2.81, 5.68, 1.03, 3.17, 1185.00},
        {14.37, 1.95, 2.50, 16.80, 113.00, 3.85, 3.49, 0.24, 2.18, 7.80, 0.86, 3.45, 1480.00},
        {13.24, 2.59, 2.87, 21.00, 118.00, 2.80, 2.69, 0.39, 1.82, 4.32, 1.04, 2.93, 735.00},
        {14.20, 1.76, 2.45, 15.20, 112.00, 3.27, 3.39, 0.34, 1.97, 6.75, 1.05, 2.85, 1450.00},
        {14.39, 1.87, 2.45, 14.60, 96.00, 2.50, 2.52, 0.30, 1.98, 5.25, 1.02, 3.58, 1290.00},
        {14.06, 2.15, 2.61, 17.60, 121.00, 2.60, 2.51, 0.31, 1.25, 5.05, 1.06, 3.58, 1295.00},
        {14.83, 1.64, 2.17, 14.00, 97.00, 2.80, 2.98, 0.29, 1.98, 5.20, 1.08, 2.85, 1045.00},
        {13.86, 1.35, 2.27, 16.00, 98.00, 2.98, 3.15, 0.22, 1.85, 7.22, 1.01, 3.55, 1045.00}
};

uint8_t y_test[10] = {2, 2, 1, 0, 0, 2, 0, 0, 1, 1};


void setup() {
    Serial.begin(115200);
    tf.begin(wine_model);

    // check if model loaded fine
    if (!tf.isOk()) {
        Serial.print("ERROR: ");
        Serial.println(tf.getErrorMessage());

        while (true) delay(1000);
    }
}

void loop() {
    for (uint8_t i = 0; i < 10; i++) {
        Serial.print("Sample #");
        Serial.print(i + 1);
        Serial.print(": ");
        Serial.print("predicted ");
        Serial.print(tf.predictClass(X_test[i]));
        Serial.print(" vs ");
        Serial.print(y_test[i]);
        Serial.println(" actual");
    }

    delay(10000);
}
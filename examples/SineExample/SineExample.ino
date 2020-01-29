#include <EloquentTinyML.h>
#include "sine_model.h"

#define NUMBER_OF_INPUTS 1
#define NUMBER_OF_OUTPUTS 1
#define TENSOR_ARENA_SIZE 2*1024

Eloquent::TinyML::TinyML<
        NUMBER_OF_INPUTS,
        NUMBER_OF_OUTPUTS,
        TENSOR_ARENA_SIZE> ml(sine_model_quantized_tflite);


void setup() {
    Serial.begin(115200);
}

void loop() {
    float x = 3.14 * random(100) / 100;
    float y = sin(x);
    float input[1] = { x };
    float predicted = ml.predict(input);

    Serial.print("sin(");
    Serial.print(x);
    Serial.print(") = ");
    Serial.print(y);
    Serial.print("\t predicted: ");
    Serial.println(predicted);
    delay(1000);
}
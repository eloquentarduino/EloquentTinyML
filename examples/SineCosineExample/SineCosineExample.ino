#include <EloquentTinyML.h>
#include "sine_cosine_model.h"

#define NUMBER_OF_INPUTS 1
#define NUMBER_OF_OUTPUTS 2
// in future projects you may need to tweek this value: it's a trial and error process
#define TENSOR_ARENA_SIZE 4*1024

Eloquent::TinyML::TfLite<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> ml;


void setup() {
    Serial.begin(115200);
    ml.begin(model_data);
}

void loop() {
    // pick up a random x and predict its sine
    float x = 3.14 * random(100) / 100;
    float input[1] = { x };
    float output[2] = {0, 0};

    ml.predict(input, output);

    Serial.print("sin(");
    Serial.print(x);
    Serial.print(") = ");
    Serial.print(sin(x));
    Serial.print("\t predicted: ");
    Serial.println(output[0]);
    Serial.print("cos(");
    Serial.print(x);
    Serial.print(") = ");
    Serial.print(cos(x));
    Serial.print("\t predicted: ");
    Serial.println(output[1]);
    delay(1000);
}
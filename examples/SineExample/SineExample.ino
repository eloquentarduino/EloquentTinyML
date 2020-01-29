#include <EloquentTinyML.h>
#include "sine_model.h"


Eloquent::TinyML::TinyML<1, 1, 2048> ml(sine_model_quantized_tflite);


void setup() {
    Serial.begin(115200);
}

void loop() {
    float input[1] = {random(10) > 5 ? 3.14/2 : 0};
    float output = ml.predict(input);

    Serial.println(output);
    delay(1000);
}
# EloquentTinyML

This Arduino library is here to simplify the deployment of Tensorflow Lite
for Microcontrollers models to Arduino boards using the Arduino IDE.

Including all the required files for you, the library exposes an eloquent
interface to load a model and run inferences.

## Install

Clone this repo in you Arduino libraries folder.

```bash
git clone https://github.com/eloquentarduino/EloquentTinyML.git
```


## Use

```cpp
#include <EloquentTinyML.h>
#include "sine_model.h"

#define NUMBER_OF_INPUTS 1
#define NUMBER_OF_OUTPUTS 1
#define TENSOR_ARENA_SIZE 2*1024

Eloquent::TinyML::TinyML<
    NUMBER_OF_INPUTS,
    NUMBER_OF_OUTPUTS,
    TENSOR_ARENA_SIZE> ml;


void setup() {
    Serial.begin(115200);
    ml.begin(sine_model_quantized_tflite);
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
```
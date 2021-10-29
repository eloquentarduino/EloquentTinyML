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

## Export TensorFlow Lite model

To run a model on your microcontroller, you should first have a model.

I suggest you use [`tinymlgen`](https://github.com/eloquentarduino/tinymlgen) to complete this step:
it will export your TensorFlow Lite model to a C array ready to be loaded
by this library.


## Use

```cpp
#include <EloquentTinyML.h>
#include "sine_model.h"

#define NUMBER_OF_INPUTS 1
#define NUMBER_OF_OUTPUTS 1
#define TENSOR_ARENA_SIZE 2*1024

Eloquent::TinyML::TfLite<
    NUMBER_OF_INPUTS,
    NUMBER_OF_OUTPUTS,
    TENSOR_ARENA_SIZE> ml;


void setup() {
    Serial.begin(115200);
    ml.begin(sine_model);
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

## Compatibility

Latest version of this library (2.4.0) is compatible with Cortex-M and ESP32 chips and is built starting from:

 - [Arduino_TensorFlowLite library version 2.4.0-ALPHA](https://www.tensorflow.org/lite/microcontrollers/overview)
 - [TensorFlowLite_ESP32 version 0.9.0](https://github.com/tanakamasayuki/Arduino_TensorFlowLite_ESP32)

ESP32 support is stuck at TensorFlow 2.1.1 at the moment.
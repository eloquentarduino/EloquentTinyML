# EloquentTinyML

This Arduino library is here to simplify the deployment of Tensorflow Lite
for Microcontrollers models to Arduino boards using the Arduino IDE.

The library exposes an *eloquent* interface to load a model and run inferences.

## Install

Install the latest version (`>=3.0.0`) from the Arduino IDE Library Manager.

You will also need `tflm_esp32` or `tflm_cortexm`, depending on your board.


## Use

```cpp
/**
 * Run a TensorFlow model to predict the IRIS dataset
 * For a complete guide, visit
 * https://eloquentarduino.com/tensorflow-lite-esp32
 */
// replace with your own model
// include BEFORE <eloquent_tinyml.h>!
#include "irisModel.h"
// include the runtime specific for your board
// either tflm_esp32 or tflm_cortexm
#include <tflm_esp32.h>
// now you can include the eloquent tinyml wrapper
#include <eloquent_tinyml.h>

// this is trial-and-error process
// when developing a new model, start with a high value
// (e.g. 10000), then decrease until the model stops
// working as expected
#define ARENA_SIZE 2000

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf;

/**
 * 
 */
void setup() {
    Serial.begin(115200);
    delay(3000);
    Serial.println("__TENSORFLOW IRIS__");
    
    // configure input/output
    // (not mandatory if you generated the .h model
    // using the everywhereml Python package)
    tf.setNumInputs(4);
    tf.setNumOutputs(3);
    // add required ops
    // (not mandatory if you generated the .h model
    // using the everywhereml Python package)
    tf.resolver.AddFullyConnected();
    tf.resolver.AddSoftmax();
    
    while (!tf.begin(irisModel).isOk())
        Serial.println(tf.exception.toString());
}


void loop() {
    // x0, x1, x2 are defined in the irisModel.h file
    // https://github.com/eloquentarduino/EloquentTinyML/tree/main/examples/IrisExample/irisModel.h
    
    // classify sample from class 0
    if (!tf.predict(x0).isOk()) {
        Serial.println(tf.exception.toString());
        return;
    }
    
    Serial.print("expcted class 0, predicted class ");
    Serial.println(tf.classification);
    
    // classify sample from class 1
    if (!tf.predict(x1).isOk()) {
        Serial.println(tf.exception.toString());
        return;
    }
    
    Serial.print("expcted class 1, predicted class ");
    Serial.println(tf.classification);
    
    // classify sample from class 2
    if (!tf.predict(x2).isOk()) {
        Serial.println(tf.exception.toString());
        return;
    }
    
    Serial.print("expcted class 2, predicted class ");
    Serial.println(tf.classification);
    
    // how long does it take to run a single prediction?
    Serial.print("It takes ");
    Serial.print(tf.benchmark.microseconds());
    Serial.println("us for a single prediction");
    
    delay(1000);
}
```
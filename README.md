# EloquentTinyML

This Arduino library is here to simplify the deployment of Tensorflow Lite
for Microcontrollers models to Arduino boards using the Arduino IDE.

Including all the required files for you, the library exposes an eloquent
interface to load a model and run inferences.

## Install

EloquentTinyML is available from the Arduino IDE Library Manager or
you can clone this repo in you Arduino libraries folder.

```bash
git clone https://github.com/eloquentarduino/EloquentTinyML.git
```

**Be sure you install version 2.4.0 or newer.**

## Export TensorFlow Lite model

To run a model on your microcontroller, you should first have a model.

I suggest you use [`tinymlgen`](https://github.com/eloquentarduino/tinymlgen) to complete this step:
it will export your TensorFlow Lite model to a C array ready to be loaded
by this library.

```python
from tinymlgen import port


tf_model = create_tf_network()
print(port(tf_model))
```


## Use

```cpp
#include <EloquentTinyML.h>
#include <eloquent_tinyml/tensorflow.h>

// sine_model.h contains the array you exported from Python with xxd or tinymlgen
#include "sine_model.h"

#define N_INPUTS 1
#define N_OUTPUTS 1
// in future projects you may need to tweak this value: it's a trial and error process
#define TENSOR_ARENA_SIZE 2*1024

Eloquent::TinyML::TensorFlow::TensorFlow<N_INPUTS, N_OUTPUTS, TENSOR_ARENA_SIZE> tf;


void setup() {
    Serial.begin(115200);
    delay(4000);
    tf.begin(sine_model);
    
    // check if model loaded fine
    if (!tf.isOk()) {
        Serial.print("ERROR: ");
        Serial.println(tf.getErrorMessage());
        
        while (true) delay(1000);
    }
}

void loop() {
    for (float i = 0; i < 10; i++) {
        // pick x from 0 to PI
        float x = 3.14 * i / 10;
        float y = sin(x);
        float input[1] = { x };
        float predicted = tf.predict(input);
        
        Serial.print("sin(");
        Serial.print(x);
        Serial.print(") = ");
        Serial.print(y);
        Serial.print("\t predicted: ");
        Serial.println(predicted);
    }

    delay(10000);
}

```

## Compatibility

Latest version of this library (2.4.0) is compatible with Cortex-M and ESP32 chips and is built starting from:

 - [Arduino_TensorFlowLite library version 2.4.0-ALPHA](https://www.tensorflow.org/lite/microcontrollers/overview)
 - [TensorFlowLite_ESP32 version 0.9.0](https://github.com/tanakamasayuki/Arduino_TensorFlowLite_ESP32)

ESP32 support is stuck at TensorFlow 2.1.1 at the moment.
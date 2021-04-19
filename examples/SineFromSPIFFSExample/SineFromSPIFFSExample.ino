#include <FS.h>
#include <SPIFFS.h>
#include <EloquentTinyML.h>
#include "sine_model.h"

#define NUMBER_OF_INPUTS 1
#define NUMBER_OF_OUTPUTS 1
// in future projects you may need to tweek this value: it's a trial and error process
#define TENSOR_ARENA_SIZE 2*1024

uint8_t *loadedModel;
Eloquent::TinyML::TfLite<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> ml;


void setup() {
    Serial.begin(115200);
    SPIFFS.begin(true);
    delay(3000);

    storeModel();
    loadModel();

    if (!ml.begin(loadedModel)) {
      Serial.println("Cannot inialize model");
      Serial.println(ml.errorMessage());
      delay(60000);
    }
}

void loop() {
    // pick up a random x and predict its sine
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

/**
 * Save model to SPIFFS
 */
void storeModel() {
  File file = SPIFFS.open("/sine.bin", "wb");

  file.write(sine_model, sine_model_len);
  file.close();
}


/**
 * Load model from SPIFFS
 */
void loadModel() {
  File file = SPIFFS.open("/sine.bin");
  size_t modelSize = file.size();

  Serial.print("Found model on filesystem of size ");
  Serial.print(modelSize);
  Serial.print(": it should be ");
  Serial.println(sine_model_len);

  // allocate memory
  loadedModel = (uint8_t*) malloc(modelSize);

  // copy data from file
  for (size_t i = 0; i < modelSize; i++)
    loadedModel[i] = file.read();
  
  file.close();
}

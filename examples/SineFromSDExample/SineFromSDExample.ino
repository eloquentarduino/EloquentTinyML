#include <SPI.h>
#include <SD.h>
#include <EloquentTinyML.h>

#define NUMBER_OF_INPUTS 1
#define NUMBER_OF_OUTPUTS 1
#define TENSOR_ARENA_SIZE 2*1024

uint8_t *loadedModel;
Eloquent::TinyML::TfLite<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> ml;


void loadModel(void);


/**
 *
 */
void setup() {
    Serial.begin(115200);
    SPI.begin();
    delay(3000);

    if (!SD.begin(4)) {
        Serial.println("Cannot init SD");
        delay(60000);
    }

    loadModel();

    if (!ml.begin(loadedModel)) {
        Serial.println("Cannot inialize model");
        Serial.println(ml.errorMessage());
        delay(60000);
    }
}


/**
 *
 */
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
 * Load model from SD
 */
void loadModel() {
    File file = SD.open("/sine.bin", FILE_READ);
    size_t modelSize = file.size();

    Serial.print("Found model on filesystem of size ");
    Serial.println(modelSize);

    // allocate memory
    loadedModel = (uint8_t*) malloc(modelSize);

    // copy data from file
    for (size_t i = 0; i < modelSize; i++)
        loadedModel[i] = file.read();

    file.close();
}
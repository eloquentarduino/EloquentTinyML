#include <SPI.h>
#include <WiFiNINA.h>
// use WiFi.h when using an ESP32
// #include <WiFi.h>
#include <HttpClient.h>
#include <EloquentTinyML.h>

#define NUMBER_OF_INPUTS 1
#define NUMBER_OF_OUTPUTS 1
#define TENSOR_ARENA_SIZE 2*1024

char SSID[] = "NetworkSSID";
char PASS[] = "Password";

// this is a server I owe that doesn't require HTTPS, you can replace with whatever server you have
const char server[] = "152.228.173.213";
const char path[] = "/sine.bin";

WiFiClient client;
HttpClient http(client);

uint8_t *model;
Eloquent::TinyML::TfLite<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> ml;



void setup() {
    Serial.begin(115200);
    delay(2000);

    wifi_connect();
    http_get();

    // init Tf from loaded model
    if (!ml.begin(model)) {
        Serial.println("Cannot inialize model");
        Serial.println(ml.errorMessage());
        delay(60000);
    }
    else {
        Serial.println("Model loaded, starting inference");
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
 * Connect to wifi
 */
void wifi_connect() {
    int status = WL_IDLE_STATUS;

    while (status != WL_CONNECTED) {
        Serial.print("Attempting to connect to SSID: ");
        Serial.println(SSID);
        status = WiFi.begin(SSID, PASS);

        delay(1000);
    }

    Serial.println("Connected to wifi");
}


/**
 * Download model from URL
 */
void http_get() {
    http.get(server, path);
    http.responseStatusCode();
    http.skipResponseHeaders();

    int modelSize = http.contentLength();

    Serial.print("Model size is: ");
    Serial.println(modelSize);
    Serial.println();

    model = (uint8_t*) malloc(modelSize);

    // copy model from response
    for (uint16_t i = 0; i < modelSize; i++)
        model[i] = http.read();

    print_model(modelSize);
}


/**
 * Dump model content
 */
void print_model(int modelSize) {
    Serial.print("Model content: ");

    for (int i = 0; i < 20; i++) {
        Serial.print(model[i], HEX);
        Serial.print(' ');
    }

    Serial.print(" ... ");

    for (int i = modelSize - 20; i < modelSize; i++) {
        Serial.print(model[i], HEX);
        Serial.print(' ');
    }

    Serial.println();
}
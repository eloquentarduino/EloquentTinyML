#define CAMERA_MODEL_M5STACK_WIDE
#include <EloquentVision.h>

Eloquent::Vision::ESP32Camera camera;
camera_fb_t *frame;



/**
 * Configure camera
 */
void initCamera() {
  camera.begin(FRAMESIZE_QVGA, PIXFORMAT_GRAYSCALE, 20000000);
}


/**
 * Capture frame from ESP32 camera
 */
uint8_t* captureFrame() {
  frame = camera.capture();

  return frame->buf;
}

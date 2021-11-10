#include "camera.h"

CameraClass cam;
uint8_t frame[320*240];


/**
 * Configure camera
 */
void initCamera() {
  cam.begin(CAMERA_R320x240, 30);
}


/**
 * Capture frame from Vision shield
 */
uint8_t* captureFrame() {
  cam.grab(frame);

  return frame;
}

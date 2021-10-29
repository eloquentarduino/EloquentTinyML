#pragma once

#include <Arduino.h>
#include <math.h>

#ifdef max
#define REDEFINE_MAX
#undef max
#undef min
#endif


#if defined(ESP32)
#include "TfLiteESP32.h"
#else
#include "TfLiteARM.h"
#endif

#ifdef REDEFINE_MAX
#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))
#endif

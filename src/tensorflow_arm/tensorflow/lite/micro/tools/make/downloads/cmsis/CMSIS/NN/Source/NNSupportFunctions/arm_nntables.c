#if !defined(ESP32)
/*
 * Copyright (C) 2010-2018 Arm Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nntables.c
 * Description:  Converts the elements of the Q7 vector to Q15 vector without left-shift
 *
 * $Date:        17. January 2018
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "tensorflow_arm/tensorflow/lite/micro/tools/make/downloads/cmsis/CMSIS/NN/Include/arm_nnsupportfunctions.h"

/**
 * @brief tables for various activation functions
 *
 * This file include the declaration of common tables.
 * Most of them are used for activation functions 
 *
 * Assumption:
 * Unified table: input is 3.x format, i.e, range of [-8, 8)
 * sigmoid(8) = 0.9996646498695336
 * tanh(8) = 0.9999997749296758
 * The accuracy here should be good enough
 *
 * 2-stage HL table: 
 *
 * The entire input range is divided into two parts:
 *
 * Low range table: 0x000x xxxx or 0x111x xxxx 
 * table entry will be the binary number excluding the first
 * two digits, i.e., 0x0x xxxx or 0x1x xxxx
 * 
 *
 *
 * High range table 0x0010 0000 -- 0x0111 1111
 *                  0x1000 0000 -- 0x1101 1111
 * 
 * For positive numbers, table entry will be
 * 0x0010 0000 -- 0x0111 1111 minus 0x0010 0000
 * i.e., 0x0000 0000 - 0x0101 11111
 *
 * same thing for the negative numbers, table entry will be
 * 0x1000 0000 -- 0x1101 1111 minux 0x0010 0000
 * i.e., 0x0110 0000 - 0x1011 1111
 */

const q7_t sigmoidTable_q7[256] = {
    0x40, 0x42, 0x44, 0x46, 0x48, 0x4a, 0x4c, 0x4e,
    0x50, 0x52, 0x53, 0x55, 0x57, 0x59, 0x5a, 0x5c,
    0x5e, 0x5f, 0x61, 0x62, 0x63, 0x65, 0x66, 0x67,
    0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f, 0x70,
    0x71, 0x72, 0x72, 0x73, 0x74, 0x74, 0x75, 0x76,
    0x76, 0x77, 0x77, 0x78, 0x78, 0x79, 0x79, 0x7a,
    0x7a, 0x7a, 0x7b, 0x7b, 0x7b, 0x7c, 0x7c, 0x7c,
    0x7c, 0x7c, 0x7d, 0x7d, 0x7d, 0x7d, 0x7d, 0x7e,
    0x7e, 0x7e, 0x7e, 0x7e, 0x7e, 0x7e, 0x7e, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
    0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x04,
    0x04, 0x04, 0x04, 0x04, 0x05, 0x05, 0x05, 0x06,
    0x06, 0x06, 0x07, 0x07, 0x08, 0x08, 0x09, 0x09,
    0x0a, 0x0a, 0x0b, 0x0c, 0x0c, 0x0d, 0x0e, 0x0e,
    0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16,
    0x17, 0x19, 0x1a, 0x1b, 0x1d, 0x1e, 0x1f, 0x21,
    0x22, 0x24, 0x26, 0x27, 0x29, 0x2b, 0x2d, 0x2e,
    0x30, 0x32, 0x34, 0x36, 0x38, 0x3a, 0x3c, 0x3e,
};

const q15_t sigmoidTable_q15[256] = {
    0x4000, 0x4200, 0x43ff, 0x45fc, 0x47f5, 0x49eb, 0x4bdc, 0x4dc8,
    0x4fad, 0x518a, 0x5360, 0x552c, 0x56ef, 0x58a8, 0x5a57, 0x5bfb,
    0x5d93, 0x5f20, 0x60a1, 0x6216, 0x637f, 0x64db, 0x662b, 0x676f,
    0x68a6, 0x69d2, 0x6af1, 0x6c05, 0x6d0d, 0x6e09, 0x6efb, 0x6fe2,
    0x70be, 0x7190, 0x7258, 0x7316, 0x73cc, 0x7478, 0x751b, 0x75b7,
    0x764a, 0x76d6, 0x775b, 0x77d8, 0x784f, 0x78c0, 0x792a, 0x798f,
    0x79ee, 0x7a48, 0x7a9d, 0x7aed, 0x7b39, 0x7b80, 0x7bc4, 0x7c03,
    0x7c3f, 0x7c78, 0x7cad, 0x7ce0, 0x7d0f, 0x7d3c, 0x7d66, 0x7d8d,
    0x7db3, 0x7dd6, 0x7df7, 0x7e16, 0x7e33, 0x7e4f, 0x7e69, 0x7e81,
    0x7e98, 0x7eae, 0x7ec2, 0x7ed5, 0x7ee7, 0x7ef8, 0x7f08, 0x7f17,
    0x7f25, 0x7f32, 0x7f3e, 0x7f4a, 0x7f55, 0x7f5f, 0x7f69, 0x7f72,
    0x7f7b, 0x7f83, 0x7f8a, 0x7f91, 0x7f98, 0x7f9e, 0x7fa4, 0x7faa,
    0x7faf, 0x7fb4, 0x7fb8, 0x7fbd, 0x7fc1, 0x7fc5, 0x7fc8, 0x7fcc,
    0x7fcf, 0x7fd2, 0x7fd5, 0x7fd7, 0x7fda, 0x7fdc, 0x7fde, 0x7fe0,
    0x7fe2, 0x7fe4, 0x7fe6, 0x7fe7, 0x7fe9, 0x7fea, 0x7feb, 0x7fed,
    0x7fee, 0x7fef, 0x7ff0, 0x7ff1, 0x7ff2, 0x7ff3, 0x7ff4, 0x7ff4,
    0x000b, 0x000c, 0x000c, 0x000d, 0x000e, 0x000f, 0x0010, 0x0011,
    0x0012, 0x0013, 0x0015, 0x0016, 0x0017, 0x0019, 0x001a, 0x001c,
    0x001e, 0x0020, 0x0022, 0x0024, 0x0026, 0x0029, 0x002b, 0x002e,
    0x0031, 0x0034, 0x0038, 0x003b, 0x003f, 0x0043, 0x0048, 0x004c,
    0x0051, 0x0056, 0x005c, 0x0062, 0x0068, 0x006f, 0x0076, 0x007d,
    0x0085, 0x008e, 0x0097, 0x00a1, 0x00ab, 0x00b6, 0x00c2, 0x00ce,
    0x00db, 0x00e9, 0x00f8, 0x0108, 0x0119, 0x012b, 0x013e, 0x0152,
    0x0168, 0x017f, 0x0197, 0x01b1, 0x01cd, 0x01ea, 0x0209, 0x022a,
    0x024d, 0x0273, 0x029a, 0x02c4, 0x02f1, 0x0320, 0x0353, 0x0388,
    0x03c1, 0x03fd, 0x043c, 0x0480, 0x04c7, 0x0513, 0x0563, 0x05b8,
    0x0612, 0x0671, 0x06d6, 0x0740, 0x07b1, 0x0828, 0x08a5, 0x092a,
    0x09b6, 0x0a49, 0x0ae5, 0x0b88, 0x0c34, 0x0cea, 0x0da8, 0x0e70,
    0x0f42, 0x101e, 0x1105, 0x11f7, 0x12f3, 0x13fb, 0x150f, 0x162e,
    0x175a, 0x1891, 0x19d5, 0x1b25, 0x1c81, 0x1dea, 0x1f5f, 0x20e0,
    0x226d, 0x2405, 0x25a9, 0x2758, 0x2911, 0x2ad4, 0x2ca0, 0x2e76,
    0x3053, 0x3238, 0x3424, 0x3615, 0x380b, 0x3a04, 0x3c01, 0x3e00,
};

const q15_t sigmoidLTable_q15[128] = {
    0x4000, 0x4100, 0x4200, 0x42ff, 0x43ff, 0x44fd, 0x45fc, 0x46f9,
    0x47f5, 0x48f1, 0x49eb, 0x4ae5, 0x4bdc, 0x4cd3, 0x4dc8, 0x4ebb,
    0x4fad, 0x509c, 0x518a, 0x5276, 0x5360, 0x5447, 0x552c, 0x560f,
    0x56ef, 0x57cd, 0x58a8, 0x5981, 0x5a57, 0x5b2a, 0x5bfb, 0x5cc9,
    0x5d93, 0x5e5b, 0x5f20, 0x5fe2, 0x60a1, 0x615d, 0x6216, 0x62cc,
    0x637f, 0x642e, 0x64db, 0x6584, 0x662b, 0x66ce, 0x676f, 0x680c,
    0x68a6, 0x693d, 0x69d2, 0x6a63, 0x6af1, 0x6b7c, 0x6c05, 0x6c8a,
    0x6d0d, 0x6d8d, 0x6e09, 0x6e84, 0x6efb, 0x6f70, 0x6fe2, 0x7051,
    0x0f42, 0x0faf, 0x101e, 0x1090, 0x1105, 0x117c, 0x11f7, 0x1273,
    0x12f3, 0x1376, 0x13fb, 0x1484, 0x150f, 0x159d, 0x162e, 0x16c3,
    0x175a, 0x17f4, 0x1891, 0x1932, 0x19d5, 0x1a7c, 0x1b25, 0x1bd2,
    0x1c81, 0x1d34, 0x1dea, 0x1ea3, 0x1f5f, 0x201e, 0x20e0, 0x21a5,
    0x226d, 0x2337, 0x2405, 0x24d6, 0x25a9, 0x267f, 0x2758, 0x2833,
    0x2911, 0x29f1, 0x2ad4, 0x2bb9, 0x2ca0, 0x2d8a, 0x2e76, 0x2f64,
    0x3053, 0x3145, 0x3238, 0x332d, 0x3424, 0x351b, 0x3615, 0x370f,
    0x380b, 0x3907, 0x3a04, 0x3b03, 0x3c01, 0x3d01, 0x3e00, 0x3f00,
};

const q15_t sigmoidHTable_q15[192] = {
    0x70be, 0x7190, 0x7258, 0x7316, 0x73cc, 0x7478, 0x751b, 0x75b7,
    0x764a, 0x76d6, 0x775b, 0x77d8, 0x784f, 0x78c0, 0x792a, 0x798f,
    0x79ee, 0x7a48, 0x7a9d, 0x7aed, 0x7b39, 0x7b80, 0x7bc4, 0x7c03,
    0x7c3f, 0x7c78, 0x7cad, 0x7ce0, 0x7d0f, 0x7d3c, 0x7d66, 0x7d8d,
    0x7db3, 0x7dd6, 0x7df7, 0x7e16, 0x7e33, 0x7e4f, 0x7e69, 0x7e81,
    0x7e98, 0x7eae, 0x7ec2, 0x7ed5, 0x7ee7, 0x7ef8, 0x7f08, 0x7f17,
    0x7f25, 0x7f32, 0x7f3e, 0x7f4a, 0x7f55, 0x7f5f, 0x7f69, 0x7f72,
    0x7f7b, 0x7f83, 0x7f8a, 0x7f91, 0x7f98, 0x7f9e, 0x7fa4, 0x7faa,
    0x7faf, 0x7fb4, 0x7fb8, 0x7fbd, 0x7fc1, 0x7fc5, 0x7fc8, 0x7fcc,
    0x7fcf, 0x7fd2, 0x7fd5, 0x7fd7, 0x7fda, 0x7fdc, 0x7fde, 0x7fe0,
    0x7fe2, 0x7fe4, 0x7fe6, 0x7fe7, 0x7fe9, 0x7fea, 0x7feb, 0x7fed,
    0x7fee, 0x7fef, 0x7ff0, 0x7ff1, 0x7ff2, 0x7ff3, 0x7ff4, 0x7ff4,
    0x000b, 0x000c, 0x000c, 0x000d, 0x000e, 0x000f, 0x0010, 0x0011,
    0x0012, 0x0013, 0x0015, 0x0016, 0x0017, 0x0019, 0x001a, 0x001c,
    0x001e, 0x0020, 0x0022, 0x0024, 0x0026, 0x0029, 0x002b, 0x002e,
    0x0031, 0x0034, 0x0038, 0x003b, 0x003f, 0x0043, 0x0048, 0x004c,
    0x0051, 0x0056, 0x005c, 0x0062, 0x0068, 0x006f, 0x0076, 0x007d,
    0x0085, 0x008e, 0x0097, 0x00a1, 0x00ab, 0x00b6, 0x00c2, 0x00ce,
    0x00db, 0x00e9, 0x00f8, 0x0108, 0x0119, 0x012b, 0x013e, 0x0152,
    0x0168, 0x017f, 0x0197, 0x01b1, 0x01cd, 0x01ea, 0x0209, 0x022a,
    0x024d, 0x0273, 0x029a, 0x02c4, 0x02f1, 0x0320, 0x0353, 0x0388,
    0x03c1, 0x03fd, 0x043c, 0x0480, 0x04c7, 0x0513, 0x0563, 0x05b8,
    0x0612, 0x0671, 0x06d6, 0x0740, 0x07b1, 0x0828, 0x08a5, 0x092a,
    0x09b6, 0x0a49, 0x0ae5, 0x0b88, 0x0c34, 0x0cea, 0x0da8, 0x0e70,
};

const q7_t tanhTable_q7[256] = {
    0x00, 0x08, 0x10, 0x18, 0x1f, 0x27, 0x2e, 0x35,
    0x3b, 0x41, 0x47, 0x4c, 0x51, 0x56, 0x5a, 0x5e,
    0x61, 0x65, 0x68, 0x6a, 0x6d, 0x6f, 0x71, 0x72,
    0x74, 0x75, 0x76, 0x78, 0x78, 0x79, 0x7a, 0x7b,
    0x7b, 0x7c, 0x7c, 0x7d, 0x7d, 0x7e, 0x7e, 0x7e,
    0x7e, 0x7e, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x81,
    0x81, 0x81, 0x81, 0x81, 0x81, 0x81, 0x81, 0x82,
    0x82, 0x82, 0x82, 0x82, 0x83, 0x83, 0x84, 0x84,
    0x85, 0x85, 0x86, 0x87, 0x88, 0x88, 0x8a, 0x8b,
    0x8c, 0x8e, 0x8f, 0x91, 0x93, 0x96, 0x98, 0x9b,
    0x9f, 0xa2, 0xa6, 0xaa, 0xaf, 0xb4, 0xb9, 0xbf,
    0xc5, 0xcb, 0xd2, 0xd9, 0xe1, 0xe8, 0xf0, 0xf8,
};

const q15_t tanhTable_q15[256] = {
    0x0000, 0x07fd, 0x0feb, 0x17b9, 0x1f59, 0x26bf, 0x2ddf, 0x34ae,
    0x3b27, 0x4142, 0x46fd, 0x4c56, 0x514d, 0x55e2, 0x5a1a, 0x5df6,
    0x617c, 0x64b0, 0x6797, 0x6a37, 0x6c95, 0x6eb5, 0x709e, 0x7254,
    0x73dc, 0x753a, 0x7672, 0x7788, 0x787f, 0x795b, 0x7a1e, 0x7acb,
    0x7b65, 0x7bee, 0x7c66, 0x7cd1, 0x7d30, 0x7d84, 0x7dce, 0x7e0f,
    0x7e49, 0x7e7d, 0x7eaa, 0x7ed2, 0x7ef5, 0x7f14, 0x7f30, 0x7f48,
    0x7f5e, 0x7f71, 0x7f82, 0x7f91, 0x7f9e, 0x7fa9, 0x7fb3, 0x7fbc,
    0x7fc4, 0x7fcb, 0x7fd1, 0x7fd7, 0x7fdc, 0x7fe0, 0x7fe4, 0x7fe7,
    0x7fea, 0x7fed, 0x7fef, 0x7ff1, 0x7ff3, 0x7ff4, 0x7ff6, 0x7ff7,
    0x7ff8, 0x7ff9, 0x7ffa, 0x7ffa, 0x7ffb, 0x7ffc, 0x7ffc, 0x7ffd,
    0x7ffd, 0x7ffd, 0x7ffe, 0x7ffe, 0x7ffe, 0x7ffe, 0x7fff, 0x7fff,
    0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff,
    0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff,
    0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff,
    0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff,
    0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff,
    0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
    0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
    0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
    0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
    0x8000, 0x8000, 0x8001, 0x8001, 0x8001, 0x8001, 0x8001, 0x8001,
    0x8001, 0x8001, 0x8001, 0x8002, 0x8002, 0x8002, 0x8002, 0x8003,
    0x8003, 0x8003, 0x8004, 0x8004, 0x8005, 0x8006, 0x8006, 0x8007,
    0x8008, 0x8009, 0x800a, 0x800c, 0x800d, 0x800f, 0x8011, 0x8013,
    0x8016, 0x8019, 0x801c, 0x8020, 0x8024, 0x8029, 0x802f, 0x8035,
    0x803c, 0x8044, 0x804d, 0x8057, 0x8062, 0x806f, 0x807e, 0x808f,
    0x80a2, 0x80b8, 0x80d0, 0x80ec, 0x810b, 0x812e, 0x8156, 0x8183,
    0x81b7, 0x81f1, 0x8232, 0x827c, 0x82d0, 0x832f, 0x839a, 0x8412,
    0x849b, 0x8535, 0x85e2, 0x86a5, 0x8781, 0x8878, 0x898e, 0x8ac6,
    0x8c24, 0x8dac, 0x8f62, 0x914b, 0x936b, 0x95c9, 0x9869, 0x9b50,
    0x9e84, 0xa20a, 0xa5e6, 0xaa1e, 0xaeb3, 0xb3aa, 0xb903, 0xbebe,
    0xc4d9, 0xcb52, 0xd221, 0xd941, 0xe0a7, 0xe847, 0xf015, 0xf803,
};

const q15_t tanhLTable_q15[128] = {
    0x0000, 0x0400, 0x07fd, 0x0bf7, 0x0feb, 0x13d7, 0x17b9, 0x1b90,
    0x1f59, 0x2314, 0x26bf, 0x2a58, 0x2ddf, 0x3151, 0x34ae, 0x37f6,
    0x3b27, 0x3e40, 0x4142, 0x442c, 0x46fd, 0x49b6, 0x4c56, 0x4edd,
    0x514d, 0x53a3, 0x55e2, 0x580a, 0x5a1a, 0x5c13, 0x5df6, 0x5fc4,
    0x617c, 0x6320, 0x64b0, 0x662d, 0x6797, 0x68f0, 0x6a37, 0x6b6e,
    0x6c95, 0x6dac, 0x6eb5, 0x6fb0, 0x709e, 0x717f, 0x7254, 0x731e,
    0x73dc, 0x7490, 0x753a, 0x75da, 0x7672, 0x7701, 0x7788, 0x7807,
    0x787f, 0x78f0, 0x795b, 0x79bf, 0x7a1e, 0x7a77, 0x7acb, 0x7b1b,
    0x849b, 0x84e5, 0x8535, 0x8589, 0x85e2, 0x8641, 0x86a5, 0x8710,
    0x8781, 0x87f9, 0x8878, 0x88ff, 0x898e, 0x8a26, 0x8ac6, 0x8b70,
    0x8c24, 0x8ce2, 0x8dac, 0x8e81, 0x8f62, 0x9050, 0x914b, 0x9254,
    0x936b, 0x9492, 0x95c9, 0x9710, 0x9869, 0x99d3, 0x9b50, 0x9ce0,
    0x9e84, 0xa03c, 0xa20a, 0xa3ed, 0xa5e6, 0xa7f6, 0xaa1e, 0xac5d,
    0xaeb3, 0xb123, 0xb3aa, 0xb64a, 0xb903, 0xbbd4, 0xbebe, 0xc1c0,
    0xc4d9, 0xc80a, 0xcb52, 0xceaf, 0xd221, 0xd5a8, 0xd941, 0xdcec,
    0xe0a7, 0xe470, 0xe847, 0xec29, 0xf015, 0xf409, 0xf803, 0xfc00,
};

const q15_t tanhHTable_q15[192] = {
    0x7b65, 0x7bee, 0x7c66, 0x7cd1, 0x7d30, 0x7d84, 0x7dce, 0x7e0f,
    0x7e49, 0x7e7d, 0x7eaa, 0x7ed2, 0x7ef5, 0x7f14, 0x7f30, 0x7f48,
    0x7f5e, 0x7f71, 0x7f82, 0x7f91, 0x7f9e, 0x7fa9, 0x7fb3, 0x7fbc,
    0x7fc4, 0x7fcb, 0x7fd1, 0x7fd7, 0x7fdc, 0x7fe0, 0x7fe4, 0x7fe7,
    0x7fea, 0x7fed, 0x7fef, 0x7ff1, 0x7ff3, 0x7ff4, 0x7ff6, 0x7ff7,
    0x7ff8, 0x7ff9, 0x7ffa, 0x7ffa, 0x7ffb, 0x7ffc, 0x7ffc, 0x7ffd,
    0x7ffd, 0x7ffd, 0x7ffe, 0x7ffe, 0x7ffe, 0x7ffe, 0x7fff, 0x7fff,
    0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff,
    0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff,
    0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff,
    0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff,
    0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff,
    0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
    0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
    0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
    0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
    0x8000, 0x8000, 0x8001, 0x8001, 0x8001, 0x8001, 0x8001, 0x8001,
    0x8001, 0x8001, 0x8001, 0x8002, 0x8002, 0x8002, 0x8002, 0x8003,
    0x8003, 0x8003, 0x8004, 0x8004, 0x8005, 0x8006, 0x8006, 0x8007,
    0x8008, 0x8009, 0x800a, 0x800c, 0x800d, 0x800f, 0x8011, 0x8013,
    0x8016, 0x8019, 0x801c, 0x8020, 0x8024, 0x8029, 0x802f, 0x8035,
    0x803c, 0x8044, 0x804d, 0x8057, 0x8062, 0x806f, 0x807e, 0x808f,
    0x80a2, 0x80b8, 0x80d0, 0x80ec, 0x810b, 0x812e, 0x8156, 0x8183,
    0x81b7, 0x81f1, 0x8232, 0x827c, 0x82d0, 0x832f, 0x839a, 0x8412,
};

#endif // end of #if !defined(ESP32)
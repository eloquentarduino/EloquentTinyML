#pragma once


/**
 *
 */
typedef float (*kernelFunction)(float *x, float *w, unsigned int dimension);

/**
 *
 * @param x
 * @param w
 * @param dimension
 * @return
 */
float linearKernel(float *x, float *w, unsigned int dimension) {
    float sum = 0;

    for(unsigned int i = 0; i < dimension; i++)
        sum += x[i] * w[i];

    return sum;
}
#pragma once

#include "kernels.h"


namespace Eloquent {
    namespace TinyML {

        /**
         *
         * @tparam D
         */
        template<unsigned int D>
        class SVMSMO {
        public:
            SVMSMO(kernelFunction kernel) :
                _kernel(kernel) {
                _params = {
                        .C = 1,
                        .tol = 1e-4,
                        .alphaTol = 1e-7,
                        .maxIter = 10000,
                        .passes = 10
                };
            }

            /**
             *
             * @param C
             */
            void setC(float C) {
                _params.C = C;
            }

            /**
             *
             * @param tol
             */
            void setTol(float tol) {
                _params.tol = tol;
            }
            
            /**
             * 
             * @param alphaTol 
             */
            void setAlphaTol(float alphaTol) {
                _params.alphaTol = alphaTol;
            }

            /**
             *
             * @param maxIter
             */
            void setMaxIter(unsigned int maxIter) {
                _params.maxIter = maxIter;
            }
            
            /**
             * 
             * @param passes 
             */
            void setPasses(unsigned int passes) {
                _params.passes = passes;
            }

            /**
             *
             * @param X
             * @param y
             * @param N num samples
             */
            void fit(float X[][D], int *y, unsigned int N) {
                _alphas = (float *) malloc(sizeof(float) * N);

                for (unsigned int i = 0; i < N; i++)
                    _alphas[i] = 0;

                unsigned int iter = 0;
                unsigned int passes = 0;

                while(passes < _params.passes && iter < _params.maxIter) {
                    float alphaChanged = 0;

                    for (unsigned int i = 0; i < N; i++) {
                        float Ei = margin(X, y, X[i], N) - y[i];

                        if ((y[i] * Ei < -_params.tol && _alphas[i] < _params.C) || (y[i] * Ei > _params.tol && _alphas[i] > 0)) {
                            // alpha_i needs updating! Pick a j to update it with
                            unsigned int j = i;

                            while (j == i)
                                j = random(0, N);

                            float Ej = margin(X, y, X[j], N) - y[j];

                            // calculate L and H bounds for j to ensure we're in [0 _params.C]x[0 _params.C] box
                            float ai = _alphas[i];
                            float aj = _alphas[j];
                            float L = 0;
                            float H = 0;

                            if (y[i] == y[j]) {
                                L = max(0, ai + aj - _params.C);
                                H = min(_params.C, ai + aj);
                            } else {
                                L = max(0, aj - ai);
                                H = min(_params.C, _params.C + aj - ai);
                            }

                            if (abs(L - H) < 1e-4)
                                continue;

                            float eta = 2 * _kernel(X[i], X[j], D) - _kernel(X[i], X[i], D) - _kernel(X[j], X[j], D);

                            if (eta >= 0)
                                continue;

                            // compute new alpha_j and clip it inside [0 _params.C]x[0 _params.C] box
                            // then compute alpha_i based on it.
                            float newaj = aj - y[j] * (Ei - Ej) / eta;

                            if (newaj > H)
                                newaj = H;
                            if (newaj < L)
                                newaj = L;
                            if (abs(aj - newaj) < 1e-4)
                                continue;

                            float newai = ai + y[i] * y[j] * (aj - newaj);

                            _alphas[i] = newai;
                            _alphas[j] = newaj;

                            // update the bias term
                            float b1 = _b - Ei - y[i] * (newai - ai) * _kernel(X[i], X[i], D)
                                       - y[j] * (newaj - aj) * _kernel(X[i], X[j], D);
                            float b2 = _b - Ej - y[i] * (newai - ai) * _kernel(X[i], X[j], D)
                                       - y[j] * (newaj - aj) * _kernel(X[j], X[j], D);

                            _b = 0.5 * (b1 + b2);

                            if (newai > 0 && newai < _params.C)
                                _b = b1;
                            if (newaj > 0 && newaj < _params.C)
                                _b = b2;

                            alphaChanged++;
                        } // end alpha_i needed updating
                    } // end for i=1..N

                    iter++;

                    if(alphaChanged == 0)
                        passes++;
                    else passes= 0;
                }

                _y = y;
                _numSamples = N;
            }

            /**
             *
             * @param x
             * @return
             */
            int predict(float X_train[][D], float x[D]) {
                return margin(X_train, _y, x, _numSamples, true) > 0 ? 1 : -1;
            }

            /**
             * Evaluate the accuracy of the classifier
             * @param X_train
             * @param X_test
             * @param y_test
             * @param testSize
             * @return
             */
            float score(float X_train[][D], float X_test[][D], int y_test[], unsigned int testSize) {
                unsigned int correct = 0;

                for (unsigned int i = 0; i < testSize; i++)
                    if (predict(X_train, X_test[i]) == y_test[i])
                        correct += 1;

                return 1.0 * correct / testSize;
            }

        protected:
            kernelFunction _kernel;
            struct {
               float C;
               float tol;
               float alphaTol;
               unsigned int maxIter;
               unsigned int passes;
            } _params;
            float _b = 0;
            unsigned int _numSamples;
            int *_y;
            float *_alphas;

            /**
             *
             * @param X
             * @param y
             * @param x
             * @param N
             * @param skipSmallAlfas
             * @return
             */
            float margin(float X[][D], int *y, float x[D], unsigned int N, bool skipSmallAlfas = false) {
                float sum = _b;

                for(unsigned int i = 0; i < N; i++)
                    if ((!skipSmallAlfas && _alphas[i] != 0) || (skipSmallAlfas && _alphas[i] > _params.alphaTol))
                        sum += _alphas[i] * y[i] * _kernel(x, X[i], D);

                return sum;
            }
        };
    }
}
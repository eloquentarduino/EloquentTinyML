#ifndef ELOQUENT_EXTRA_TIME_BENCHMARK
#define ELOQUENT_EXTRA_TIME_BENCHMARK

namespace Eloquent {
    namespace Extra {
        namespace Time {
            /**
             * Run benchmark on code blocks
             */
            class Benchmark {
            public:

                /**
                 * Start timer
                 */
                void start() {
                    timeStart = micros();
                }

                /**
                 * Stop timer
                 */
                size_t stop() {
                    elapsedInMicros = micros() - timeStart;

                    return millis();
                }

                /**
                 * Benchmark given function
                 */
                template<typename Callback>
                size_t benchmark(Callback callback) {
                    start();
                    callback();

                    return stop();
                }

                /**
                 * Get elapsed time in millis
                 */
                inline size_t millis() {
                    return microseconds() / 1000;
                }

                /**
                 * Alias for millis()
                 */
                inline size_t ms() {
                    return millis();
                }

                /**
                 * Get elapsed time in micros
                 */
                inline size_t microseconds() {
                    return elapsedInMicros;
                }

                /**
                 * Alias for micros
                 */
                inline size_t us() {
                    return microseconds();
                }

                /**
                 *
                 */
                template<typename Callback>
                void timeit(Callback callback) {
                    start();
                    callback();
                    stop();
                }

            protected:
                size_t timeStart;
                size_t elapsedInMicros;
            };
        }
    }
}

#endif

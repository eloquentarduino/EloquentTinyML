#pragma once

/**
 * Wrapper for RGB color sensor
 */
class RGB {
    public:
        RGB(uint8_t s2, uint8_t s3, uint8_t out) :
            _s2(s2),
            _s3(s3),
            _out(out) {

        }

        /**
         *
         */
        void begin() {
            pinMode(_s2, OUTPUT);
            pinMode(_s3, OUTPUT);
            pinMode(_out, INPUT);
        }

        /**
         *
         * @param x
         */
        void read(float x[3]) {
            x[0] = readComponent(LOW, LOW);
            x[1] = readComponent(HIGH, HIGH);
            x[2] = readComponent(LOW, HIGH);
        }

    protected:
        uint8_t _s2;
        uint8_t _s3;
        uint8_t _out;

        /**
         *
         * @param s2
         * @param s3
         * @return
         */
        int readComponent(bool s2, bool s3) {
            delay(10);
            digitalWrite(_s2, s2);
            digitalWrite(_s3, s3);

            return pulseIn(_out, LOW);
        }
};
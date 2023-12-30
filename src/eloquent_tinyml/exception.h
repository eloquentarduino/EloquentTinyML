#ifndef ELOQUENT_EXCEPTION_H
#define ELOQUENT_EXCEPTION_H


namespace Eloquent {
    namespace Error {
        /**
         * Application expcetion
         */
        class Exception {
            public:
                /**
                 * 
                 */
                Exception(const char* tag) : 
                    _tag(tag), 
                    _message(""),
                    _isSevere(true) {
                }

                /**
                 * Test if there's an exception
                 */
                operator bool() const {
                    return !isOk();
                }

                /**
                 * Test if there's an exception
                 */
                bool isOk() const {
                    return _message == "";
                }

                /**
                 * Test if exception is severe
                 */
                bool isSevere() const {
                    return _isSevere && !isOk();
                }

                /**
                 * Mark error as not severe
                 */
                Exception& soft() {
                    _isSevere = false;

                    return *this;
                }

                /**
                 * Set exception message
                 */
                Exception& set(String error) {
                    _message = error;
                    _isSevere = true;

                    return *this;
                }

                /**
                 * Clear exception
                 */
                Exception& clear() {
                    return set("");
                }

                /**
                 * Copy exception from other source
                 */
                template<typename Other>
                Exception& from(Other& other) {
                    set(other.exception.toString());

                    return *this;
                }

                /**
                 * Convert exception to string
                 */
                inline String toString() {
                    return _message;
                }

                /**
                 * Convert exception to char*
                 */
                inline const char* toCString() {
                    return toString().c_str();
                }

            protected:
                const char* _tag;
                bool _isSevere;
                String _message;
        };
    }
}

#endif
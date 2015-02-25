
//
// g++ -I. ann_classifier.cpp -shared -o ann.dll
//

/*
 * Wrapper for Python
 *
 *
 */

#include <cstdio>
#include <ann_base.hpp>


typedef float DATATYPE;

extern "C" {


    void* ann_fromfile(const char* fname) {
        FILE* fin = fopen(fname, "rb");
        if (!fin)
            return NULL;

        fseek(fin, 0, SEEK_END);
        size_t size = ftell(fin);
        fseek(fin, 0, SEEK_SET);

        size_t buffer_size = size / sizeof(DATATYPE);
        ma::memory::ptr_vec<DATATYPE> buffer(new DATATYPE[buffer_size]);
        size_t read = fread(buffer.get(), size, 1, fin);

        ma::ann<DATATYPE>* ann = new ma::ann<DATATYPE>(buffer.get());

        return ann;
    }

    void ann_free(void* ann) {
        delete static_cast< ma::ann<DATATYPE>* >(ann);
    }

    void ann_predict(void* ann, const DATATYPE* X, DATATYPE* predictions, int rows) {
        static_cast< ma::ann<DATATYPE>* >(ann)->predict(X, predictions, rows);
    }



}




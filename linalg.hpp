
#if !defined LINALG_DOT_HPP
#define LINALG_DOT_HPP



#include <memory.hpp>

using namespace std;


namespace ma {

namespace linalg {

    template<class T>
    void fill(T* v, int size, T val) {
        for (int i = 0; i < size; ++i)
            v[i] = val;
    }

    template <class T>
    memory::ptr_vec<T> zeros(int size) {
        memory::ptr_vec<T> tmp(new T[size]);
        fill(tmp.get(), size, (T)0);
        return tmp;
    }


    template<class T>
    void copy(T* v1, const T* v2, int size) {
        for (int i = 0; i < size; ++i) {
            v1[i] = v2[i];
        }
    }


    // scalar sum of all elements of the vector v
    // Semantic: sum(v)
    template<class T>
    T sum(const T* v, int size) {
        T accum = 0.;
        for (int i = 0; i < size; ++i)
            accum += v[i];
        return accum;
    }

    // Elementwise sum of 2 vectors
    // the result goes to v1
    // Semantic: v1 += v2
    template<class T>
    void sum_v2v(T* v1, const T* v2, int size) {
        int size_4 = size / 4;
        for (int i = 0; i < size_4; ++i) {
            v1[i*4 + 0] += v2[i*4 + 0];
            v1[i*4 + 1] += v2[i*4 + 1];
            v1[i*4 + 2] += v2[i*4 + 2];
            v1[i*4 + 3] += v2[i*4 + 3];
        }
        for (int i = size_4 * 4; i < size; ++i) {
            v1[i] += v2[i];
        }
    }

    template<class T>
    void sub_v2v(const T* v1, const T* v2, T* result, int size) {
        for (int i = 0; i < size; ++i) {
            result[i] = v1[i] - v2[i];
        }
    }

    template<class T>
    void mul_v2v(T* v1, const T* v2, int size) {
        for (int i = 0; i < size; ++i) {
            v1[i] *= v2[i];
        }
    }


    template<class T>
    void mul_add_v2v(const T* v1, const T* v2, T* result, int size) {
        int size_4 = size / 4;
        for (int i = 0; i < size_4; ++i) {
            result[i*4 + 0] += v1[i*4 + 0] * v2[i*4 + 1];
            result[i*4 + 1] += v1[i*4 + 1] * v2[i*4 + 1];
            result[i*4 + 2] += v1[i*4 + 2] * v2[i*4 + 2];
            result[i*4 + 3] += v1[i*4 + 3] * v2[i*4 + 3];
        }
        for (int i = size_4 * 4; i < size; ++i) {
            result[i] += v1[i] * v2[i];
        }
    }

    template<class T>
    void mul_add_v2s(const T* v1, T s, T* result, int size) {
        int size_4 = size / 4;
        for (int i = 0; i < size_4; ++i) {
            result[i*4 + 0] += v1[i*4 + 0] * s;
            result[i*4 + 1] += v1[i*4 + 1] * s;
            result[i*4 + 2] += v1[i*4 + 2] * s;
            result[i*4 + 3] += v1[i*4 + 3] * s;
        }
        for (int i = size_4 * 4; i < size; ++i) {
            result[i] += v1[i] * s;
        }
    }


    template<class T>
    void mul_sub_v2s(const T* v1, T s, T* result, int size) {
        int size_4 = size / 4;
        for (int i = 0; i < size_4; ++i) {
            result[i*4 + 0] -= v1[i*4 + 0] * s;
            result[i*4 + 1] -= v1[i*4 + 1] * s;
            result[i*4 + 2] -= v1[i*4 + 2] * s;
            result[i*4 + 3] -= v1[i*4 + 3] * s;
        }
        for (int i = size_4 * 4; i < size; ++i) {
            result[i] -= v1[i] * s;
        }
    }


    template<class T>
    void div_v2s(T* v, int size, T val) {
        int size_4 = size / 4;
        for (int i = 0; i < size_4; ++i) {
            v[i*4 + 0] /= val;
            v[i*4 + 1] /= val;
            v[i*4 + 2] /= val;
            v[i*4 + 3] /= val;
        }
        for (int i = size_4 * 4; i < size; ++i) {
            v[i] /= val;
        }
    }

    // Dot product of 2 vectors
    // Ssemantic: v1 . v2
    template<class T>
    T dot_v2v(const T* v1, const T* v2, int size) {
        T result = (T)0;

        int size_4 = size / 4;
        for (int i = 0; i < size_4; ++i) {
            result += v1[i*4 + 0] * v2[i*4 + 0];
            result += v1[i*4 + 1] * v2[i*4 + 1];
            result += v1[i*4 + 2] * v2[i*4 + 2];
            result += v1[i*4 + 3] * v2[i*4 + 3];
        }
        for (int i = size_4 * 4; i < size; ++i) {
            result += v1[i] * v2[i];
        }

        return result;
    }

    // Dot product of matrix and vector
    // Semantic: m . v
    template<class T>
    void dot_m2v(const T* m, const T* v, T* result, int rows, int columns) {
        for (int r = 0; r < rows; ++r) {
            result[r] = dot_v2v(v, &m[r * columns], columns);
        }
    }


    template<class T>
    void transpose(const T* m, T* result, int rows, int cols) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                result[c*rows + r] = m[r*cols + c];
            }
        }
    }

}

}

#endif

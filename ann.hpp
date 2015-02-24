

#if !defined ANN_DOT_HPP
#define ANN_DOT_HPP


#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>

#include <memory.hpp>
#include <linalg.hpp>
#include <random.hpp>


using namespace std;


namespace ma {

template <class T>
class ann_leaner {
    vector<int> sizes_;

    memory::ptr_vec<T> aa_;
    memory::ptr_vec<T> ww_;
    memory::ptr_vec<T> bb_;

    memory::ptr_vec<T> deltas_;

    memory::ptr_vec<T> bb_deriv_;
    memory::ptr_vec<T> ww_deriv_;

    int total_bb_size_;
    int total_ww_size_;
    int total_aa_size_;

    bool verbose_;


public:
    ann_leaner(const vector<int>& sizes) :
        sizes_(sizes),
        aa_(),
        ww_(),
        bb_(),
        deltas_(),
        bb_deriv_(),
        ww_deriv_(),
        total_bb_size_(0),
        total_ww_size_(0),
        total_aa_size_(0),
        verbose_(false)
    {
        int layers_num = sizes_.size();

        // calculate sizes for arrays
        // I skip level 0 as it's the input vector x
        for (int l = 1; l < layers_num; ++l) {
            int l_layer_size = sizes_[layers_num - l];

            total_bb_size_ += l_layer_size;
            total_aa_size_ += l_layer_size;
            total_ww_size_ += (l_layer_size * sizes_[layers_num - l - 1]);
        }

        // alloacate vectors
        aa_.reset(new T[total_aa_size_]);
        bb_.reset(new T[total_bb_size_]);
        ww_.reset(new T[total_ww_size_]);

        deltas_.reset(new T[total_aa_size_]);

        // partial derivatives
        bb_deriv_.reset(new T[total_bb_size_]);
        ww_deriv_.reset(new T[total_ww_size_]);

        // initialise biases and weights with random values
        random::rand<T>(bb_.get(), total_bb_size_);
        random::rand<T>(ww_.get(), total_ww_size_);

        for (int i = 0; i < total_bb_size_; ++i)
            bb_[i] -= .5;

//        for (int i = 0; i < total_ww_size_; ++i)
        int ww_idx = 0;
        for (int l = 1; l < layers_num; ++l) {
            int size = sizes_[l] * sizes_[l-1];
            for (int i = 0; i < size; ++i) {
                ww_[ww_idx + i] = (ww_[ww_idx + i] - .5); // / ::sqrt(sizes_[l-1]);
            }
            ww_idx += size;
        }

    }
/*
    ann_leaner(const T* buffer) :
        sizes_(sizes),
        aa_(),
        ww_(),
        bb_(),
        deltas_(),
        bb_deriv_(),
        ww_deriv_(),
        total_bb_size_(0),
        total_ww_size_(0),
        total_aa_size_(0),
        verbose_(false)
    {
    }
*/

    template <class I>
    void print_vector(ostream& os, const I* vec, int size, const char* comment, const char* name) {
        os << "// " << comment << endl;
        os << "int " << name << "_size = " << size << ";" << endl;
        os << "int " << name << "[] = {";
        for (int i = 0; i < size; ++i) {
            os << setprecision(16) << vec[i] << ",";
        }
        os << "};" << endl;
    }

    void print(ostream& os) {
        int layers_num = sizes_.size();

        print_vector(os, &sizes_[0], sizes_.size(), "ANN", "sizes");
        print_vector(os, bb_.get(), total_bb_size_, "biases", "bb");
        print_vector(os, ww_.get(), total_ww_size_, "weights", "ww");
        print_vector(os, bb_deriv_.get(), total_bb_size_, "bb derivs", "bb_deriv");
        print_vector(os, ww_deriv_.get(), total_ww_size_, "ww derivs", "ww_deriv");

    }

    // calculates sigmoid in place
    void sigmoid(T* v, int size) {
        for (int i = 0; i < size; ++i) {
            double val = v[i];
            v[i] = 1. / (1. + ::exp(-val));
        }
    }



    void reset_deriv_for_next_minibatch() {
        linalg::fill(bb_deriv_.get(), total_bb_size_, (T)0);
        linalg::fill(ww_deriv_.get(), total_ww_size_, (T)0);
    }

    void random_shift() {
        double v;
        for (int b = 0; b < total_bb_size_; ++b) {
            random::rand(&v, 1);
            if (.7 < v) {
                random::rand(&v, 1);
                bb_[b] = (v - .5) ;
            }
        }

        for (int w = 0; w < total_ww_size_; ++w) {
            random::rand(&v, 1);
            if (.7 < v) {
                random::rand(&v, 1);
                ww_[w] = (v - .5) ;
            }
        }
    }


    // the size of x is sizes_[0]
    void forward(const T* x) {
        int layers_num = sizes_.size();

        const T* activation = x;

        int aa_idx = 0;
        int bb_idx = 0;
        int ww_idx = 0;

        for (int l = 1; l < layers_num; ++l) {
            // matrix, vector, result-vector, rows, columns
            linalg::dot_m2v<T>(&ww_[ww_idx], activation, &aa_[aa_idx], sizes_[l], sizes_[l - 1]);
            linalg::sum_v2v<T>(&aa_[aa_idx], &bb_[bb_idx], sizes_[l]);
            sigmoid(&aa_[aa_idx], sizes_[l]);
            activation = &aa_[aa_idx];

            aa_idx += sizes_[l];
            bb_idx += sizes_[l];
            ww_idx += (sizes_[l] * sizes_[l - 1]);
        }

        if (verbose_) {
            cout << "# forward result: {";
            aa_idx -= sizes_[layers_num - 1];
            for (int i = 0; i < sizes_[layers_num - 1]; ++i) {
                cout << aa_[aa_idx + i] << ",";
            }
            cout << "}" << endl;
        }
    }

    T backward(const T* x, const T* y) {

        // accumulator for cost
        T cost = 0;

        //
        int aa_idx = total_aa_size_;
        int ww_idx = total_ww_size_;

        int layers_num = sizes_.size();

        for (int l = 1; l < layers_num; ++l) {
            int l_idx = layers_num - l;

            // define shifts in the buffers
            aa_idx -= sizes_[l_idx];
            ww_idx -= (sizes_[l_idx] * sizes_[l_idx - 1]);

            if (1 == l) {
                // last layer

                // calculate derivatives
                for (int a = 0; a < sizes_[l_idx]; ++a) { // rows
                    //cost += (aa_[aa_idx + a] - y[a]) * (aa_[aa_idx + a] - y[a]);

                    cost +=  -1. * (y[a] * ::log(aa_[aa_idx + a]) + (1. - y[a]) * ::log(1. - aa_[aa_idx + a])) ;


                    //T sig_deriv = aa_[aa_idx + a] - aa_[aa_idx + a] * aa_[aa_idx + a];

                    T delta = (aa_[aa_idx + a] - y[a]); // * sig_deriv;

                    deltas_[aa_idx + a] = delta;
                    bb_deriv_[aa_idx + a] += delta;

                    int aa_idx_prev = aa_idx - sizes_[l_idx - 1];
                    for (int a_prev = 0; a_prev < sizes_[l_idx - 1]; ++a_prev) { // columns
                        ww_deriv_[ww_idx + a * sizes_[l_idx - 1] + a_prev] += aa_[aa_idx_prev + a_prev] * delta;
                    }
                }

            }
            else {
                // hidden layers

                int aa_idx_prev = aa_idx - sizes_[l_idx - 1];
                int aa_idx_next = aa_idx + sizes_[l_idx];
                int ww_idx_next = ww_idx + (sizes_[l_idx] * sizes_[l_idx - 1]);

                // calculate derivatives
                for (int a = 0; a < sizes_[l_idx]; ++a) {
                    T sig_deriv = aa_[aa_idx + a] - aa_[aa_idx + a] * aa_[aa_idx + a];
                    T delta = 0;
                    for (int a_next = 0; a_next < sizes_[l_idx + 1]; ++a_next) {
                        delta += ww_[ww_idx_next + a_next * sizes_[l_idx] + a] * deltas_[aa_idx_next + a_next];
                    }
                    delta *= sig_deriv;
                    deltas_[aa_idx + a] = delta;
                    bb_deriv_[aa_idx + a] += delta;

                    const T* activation = aa_idx_prev >= 0 ? &aa_[aa_idx_prev] : x;

                    for (int a_prev = 0; a_prev < sizes_[l_idx - 1]; ++a_prev) {
                        ww_deriv_[ww_idx + a * sizes_[l_idx - 1] + a_prev] += activation[a_prev] * delta;
                    }
                }
            }
        }

        return cost; // / 2.;
    }

    void average_deriv(T sample_size) {
        linalg::div_v2s(bb_deriv_.get(), total_bb_size_, sample_size);
        linalg::div_v2s(ww_deriv_.get(), total_ww_size_, sample_size);
    }



    T fit_minibatch(const T* X, const T* Y, int rows, T alpha) {

        reset_deriv_for_next_minibatch();

        int x_columns_num = sizes_[0];
        int y_columns_num = sizes_[ sizes_.size() - 1 ];

        T cost = (T)0;

        for (int r = 0; r < rows; ++r) {
            forward(&X[r * x_columns_num]);
            cost += backward(&X[r * x_columns_num], &Y[r * y_columns_num]);
        }

        average_deriv((T)rows);

        // update biases and weights

        T v;

        for (int b = 0; b < total_bb_size_; ++b) {
            //random::rand(&v, 1);
            //if (.3 < v)
                bb_[b] -= alpha * bb_deriv_[b];
        }

        for (int w = 0; w < total_ww_size_; ++w) {
            //random::rand(&v, 1);
            //if (.3 < v)
                ww_[w] -= alpha * ww_deriv_[w];
        }

        return cost / rows;
    }


    void get_output(T* y) {
        int layers_num = sizes_.size();
        int aa_idx = total_aa_size_ - sizes_[layers_num - 1];
        for (int a = 0; a < sizes_[layers_num - 1]; ++a) {
            y[a] = aa_[aa_idx + a];
        }
    }


    // Helper method to calculate cost of current sample
    T cost(const T* y) {
        int layers_num = sizes_.size();
        int aa_idx = total_aa_size_ - sizes_[layers_num - 1];
        T cost = 0;
        for (int a = 0; a < sizes_[layers_num - 1]; ++a) {
            cost += (aa_[aa_idx + a] - y[a]) * (aa_[aa_idx + a] - y[a]);
        }
        return cost / 2.;
    }

    // Helper method to debug backward propagation logic
    void calc_deriv(const T* x, const T* y, memory::ptr_vec<T>& bd, memory::ptr_vec<T>& wd) {
        T epsilon = 0.0000001;

        // prepare vectors
        bd.reset(new T[total_bb_size_]);
        wd.reset(new T[total_ww_size_]);

        // biases
        for (int b = 0; b < total_bb_size_; ++b) {
            // save current bb value before changing
            T tmp_b = bb_[b];

            // 1st pass
            bb_[b] -= epsilon;
            forward(x);
            T cost_before = cost(y);
            bb_[b] = tmp_b;

            // 2nd pass
            bb_[b] += epsilon;
            forward(x);
            T cost_after = cost(y);

            // calc derivative
            T deriv = (cost_after - cost_before) / (2. * epsilon);
            bd[b] = deriv;

            // restore bb val
            bb_[b] = tmp_b;
        }

        // weights
        for (int w = 0; w < total_ww_size_; ++w) {
            // save current ww value before changing
            T tmp_w = ww_[w];

            // 1st pass
            ww_[w] -= epsilon;
            forward(x);
            T cost_before = cost(y);
            ww_[w] = tmp_w;

            // 2nd pass
            ww_[w] += epsilon;
            forward(x);
            T cost_after = cost(y);

            // calc derivative
            T deriv = (cost_after - cost_before) / (2. * epsilon);
            wd[w] = deriv;

            // restore ww val
            ww_[w] = tmp_w;
        }

    }

    const T* get_bb_deriv() const { return bb_deriv_.get(); }
    const T* get_ww_deriv() const { return ww_deriv_.get(); }

};

}



















#endif


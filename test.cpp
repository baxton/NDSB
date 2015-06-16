#include <iostream>
#include <cstdlib>
#include <time.h>
#include <limits>
#include <cmath>


using namespace std;



double mul_div(double* o, size_t size) {
    double res = 1.;

    size_t idx_o = 0;
    double idx_n = 1;

    while (idx_o < size && idx_n <= size) {
        res *= (o[idx_o] / idx_n);
        idx_o += 1;
        idx_n += 1;

        while (res > 1000000000. && idx_n <= size)
            res /= idx_n++;

        while (res < .0000000001 && idx_o < size)
            res *= o[idx_o++];
    }

    while (idx_o < size)
        res *= o[idx_o++];

    while (idx_n <= size)
        res /= idx_n++;

    return res;
}


double sum_sub(double* s, size_t size) {
    size_t idx_n = 1;

    ssize_t res = 0;
    for (size_t i = 0; i < size; ++i)
        res += ((idx_n++) - (size_t)s[i]);
    return res;
}

void shuffle(double* o, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        size_t idx = (rand() % size);
        double tmp = o[i];
        o[i] = o[idx];
        o[idx] = tmp;
    }
}

void fill(double* o, size_t size) {
    for (size_t i = 0; i < size; ++i)
        o[i] = (double)(i+1);
}




void print(ostream& os, double* o, size_t size) {
    os << "double orig[" << size << "] = {";
    for (size_t i = 0; i < size; ++i)
        os << o[i] << ",";
    os << "};" << endl;
}



int run(double* o, size_t N) {
    fill(&o[0], N);
//    print(cout, o, N);
    shuffle(o, N);
//    print(cout, o, N);


    // miss + dup
    size_t idx_missed = rand() % N;
    size_t idx_duplicated = rand() % N;
    while (idx_duplicated == idx_missed)
        idx_duplicated = rand() % N;

    double missed = o[idx_missed];
    double duplicated = o[idx_duplicated];

    o[idx_missed] = duplicated;

    cout << "Missed: " << (size_t)missed << endl
         << "Duplicated: " << (size_t)duplicated << endl;
//    print(cout, o, N);



    double k = mul_div(o, N);
    cout << "MUL_DIV: " << k << endl;

    double ss = sum_sub(o, N);


    //
    double m = round(ss / (1. - k));
    double d = round(k * m);
    int err = fabs(missed - m) + fabs(duplicated - d);

    cout << "Calculated result" << endl;
    cout << "Missed: " << (size_t)m << endl
         << "Duplicated: " << (size_t)d << endl
         << "Error: " << err << endl;


    return err;
}


int main(int argc, const char** argv) {

    srand(time(NULL));

    if (argc > 1) {
        size_t N = atoi(argv[1]);
        cout << "N = " << N << endl;

        double* o = new double[N];
        int err = run(o, N);
        delete [] o;

    }
    else {
        size_t N = 500000000;
        double* o = new double[N];

        for (size_t n = N; n > 1; --n) {
            cout << "N = " << n << endl;
            for (int t = 0; t < 10; ++t) {
                int err = run(o, n);
                if (err) {
                    // todo
                }
            }
        }
        delete [] o;
    }

    return 0;
}



#include <cmath>
#include <iostream>
#include <vector>

#include <random.hpp>
#include <ann.hpp>


int main() {
    ma::random::seed();

    vector<int> sizes;
    sizes.push_back(3);
    sizes.push_back(10);
    sizes.push_back(5);
    ma::ann_leaner<double> nn(sizes);

    nn.reset_deriv_for_next_minibatch();

    double x[] = {1,2,3,  2,5,3};
    double y[] = {0, 0, 0, 1, 0,  0, 0, 1, 0, 0};

/*
    nn.forward(&x[0]);
    double cost = nn.backward(&x[0], &y[0]);
    nn.forward(&x[3]);
    cost += nn.backward(&x[3], &y[5]);
    nn.average_deriv(2.);
*/

    double cost = nn.fit_minibatch(x, y, 2, .2);

    nn.print(cout);
    cout << "Cost: " << (cost / 2.) << endl;

    //
    int total_bd_size = 10 + 5;
    int total_wd_size = 3*10 + 10*5;

    ma::memory::ptr_vec<double> bd_accum(new double[total_bd_size]);
    ma::memory::ptr_vec<double> wd_accum(new double[total_wd_size]);
    ma::linalg::fill(bd_accum.get(), total_bd_size, (double)0);
    ma::linalg::fill(wd_accum.get(), total_wd_size, (double)0);

    {
        ma::memory::ptr_vec<double> bd, wd;

        nn.calc_deriv(&x[0], &y[0], bd, wd);
        ma::linalg::sum_v2v(bd_accum.get(), bd.get(), total_bd_size);
        ma::linalg::sum_v2v(wd_accum.get(), wd.get(), total_wd_size);

        nn.calc_deriv(&x[3], &y[5], bd, wd);
        ma::linalg::sum_v2v(bd_accum.get(), bd.get(), total_bd_size);
        ma::linalg::sum_v2v(wd_accum.get(), wd.get(), total_wd_size);

        ma::linalg::div_v2s(bd_accum.get(), total_bd_size, 2.);
        ma::linalg::div_v2s(wd_accum.get(), total_wd_size, 2.);

    }


    cout << "BD vs BB_DERIV" << endl;
    for (int b = 0; b < total_bd_size; ++b)
        cout << bd_accum[b] << "\t" << nn.get_bb_deriv()[b] << endl;

    cout << "WD vs WW_DERIV" << endl;
    for (int w = 0; w < total_wd_size; ++w)
        cout << wd_accum[w] << "\t" << nn.get_ww_deriv()[w] << endl;

    double b_error = 0;
    double w_error = 0;

    for (int b = 0; b < total_bd_size; ++b)
        b_error += ::sqrt((bd_accum[b] - nn.get_bb_deriv()[b]) * (bd_accum[b] - nn.get_bb_deriv()[b]));

    for (int w = 0; w < total_wd_size; ++w)
        w_error += ::sqrt((wd_accum[w] - nn.get_ww_deriv()[w]) * (wd_accum[w] - nn.get_ww_deriv()[w]));

    cout << "BD avr error: " << (b_error / total_bd_size) << endl;
    cout << "WD avr error: " << (w_error / total_wd_size) << endl;

    // test for decreasing cost
    double alpha = 7.6;
    for (int i = 0; i < 10000; ++i) {
        cost = nn.fit_minibatch(x, y, 2, alpha);
        if (0 == (i % 500)) {
            cout << "Cost: " << cost << endl;
            //alpha -= .01;
        }
    }
    cout << "Cost: " << cost << endl;



    return 0;
}

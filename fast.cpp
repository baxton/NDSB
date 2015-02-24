

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <map>
#include <iomanip>

#include <random.hpp>
#include <ann.hpp>
#include <id_file_map.h>
#include <train_indices.h>



using namespace std;


//#define SHORT_SET

typedef float DATATYPE;


const int ALIGN = 512;

const int VEC_LEN = 1 + 2 + 48 * 48;
const int FILE_SIZE_BYTES = 9228;
const int FILE_SIZE = FILE_SIZE_BYTES / sizeof(DATATYPE);
const int CLS_NUM = 121;



void get_file_id_map(map<int, pair<string, string> >& file_id_map) {
    int id;
    char folder[128];
    char fname[128];

    for (int f = 0; f < ma::train_ids_files_size; ++f) {
        ifstream fin(ma::train_ids_files[f], ifstream::in);
        string line;
        while (std::getline(fin, line)) {
            size_t pos = line.find("\\\\");
            line.replace(pos, 2, " ");
            pos = line.find("\"");
            line.replace(pos, 1, "");
            pos = line.find("\"");
            line.replace(pos, 1, "");
            sscanf(line.c_str(), "%d %s %s", &id, folder, fname);

            //cout << "line: " << id << "; " << folder << "; " << fname << endl;

            file_id_map[id] = pair<string, string>(folder, fname);
        }
    }
}


void fill_in_mini_batch(int* mini_batch_ids, int mini_batch_size, int train_files_num, DATATYPE* buffer, DATATYPE* y, map<int, pair<string, string> >& file_id_map, const string& path_train) {

#if !defined SHORT_SET
    ma::random::get_k_of_n(mini_batch_size, train_files_num, mini_batch_ids);
#else
    ma::random::get_k_of_n(mini_batch_size, train_indices_size, mini_batch_ids);
#endif

    int row = 0;

    ma::memory::ptr_vec<DATATYPE> local_buf(new DATATYPE[FILE_SIZE]);

    for (int i = 0; i < mini_batch_size; ++i) {
#if defined SHORT_SET
        int idx_idx = mini_batch_ids[i];
        int idx = train_indices[idx_idx];
#else
        int idx = mini_batch_ids[i];
#endif

        pair<string, string>& p = file_id_map[ idx ];
#if defined FOR_LINUX
        string fname = path_train + p.first + "/" + p.second + ".b";
#else
        string fname = path_train + p.first + "\\" + p.second + ".b";
#endif

        FILE* fin = fopen(fname.c_str(), "rb");

        // get file size
        fseek(fin, 0, SEEK_END);
        size_t file_size = ftell(fin);
        fseek(fin, 0, SEEK_SET);

        // sanity check
        if (file_size != FILE_SIZE_BYTES)
            cout << "# ERROR file size is different " << file_size << " vs " << FILE_SIZE_BYTES << endl;

        // read
        size_t read = fread(local_buf.get(), file_size, 1, fin);

        for (int cls_idx = 0; cls_idx < CLS_NUM; ++cls_idx)
            y[row * CLS_NUM + cls_idx] = 0.;
        y[row * CLS_NUM + int(local_buf[0])] = 1.;


        //cout << "# ID " << mini_batch_ids[i] << " cls " << int(local_buf[0]) << " (" << fname << ") " << read << endl;

        // this does not include the 1st item which is a class
        int data_vec_len = VEC_LEN - 1;
        for (int val_idx = 0; val_idx < data_vec_len; ++val_idx)
            buffer[row * data_vec_len + val_idx] = local_buf[1 + val_idx];

        row += 1;


        //cout << "# loaded [" << mini_batch_ids[i] << "]: " << fname << " (" << file_size << ")" << endl;

        fclose(fin);
    }
}




int main() {

    map<int, pair<string, string> > file_id_map;
    get_file_id_map(file_id_map);

#if defined FOR_LINUX
    const string path_train = "/home/maxim/kaggle/NDSB/data/train_ann1/";
    const char* path_tmp = "/home/maxim/kaggle/NDSB/tmp/";
#else
    cout << "# built for Windows" << endl;
    const string path_train = "C:\\Temp\\kaggle\\NDSB\\data\\train_ann1\\";
    const char* path_tmp = "C:\\Temp\\kaggle\\NDSB\\tmp\\";
#endif


    const int train_files_num = 30336;

#if defined SHORT_SET
    const int mini_batch_size = 1585;
#else
    const int mini_batch_size = 30336;
#endif

    int mini_batch_ids[mini_batch_size];

    // allocate mini-batch buffer
    const int vector_len = VEC_LEN - 1; // remove class
    ma::memory::ptr_vec<DATATYPE> buffer_non_aligned(new DATATYPE[ ALIGN + mini_batch_size * vector_len ]);
    size_t p = (size_t)buffer_non_aligned.get();
    DATATYPE* buffer =  (DATATYPE*)((p + ALIGN) - (p % ALIGN));

    ma::memory::ptr_vec<DATATYPE> Y (new DATATYPE[mini_batch_size * CLS_NUM]);


    // learn
    ma::random::seed();


    vector<int> sizes;
    sizes.push_back(VEC_LEN - 1);
    sizes.push_back(100);
    //sizes.push_back(50);
    //sizes.push_back(4);
    //sizes.push_back(2);
    //sizes.push_back(4);
    //sizes.push_back(2);
    sizes.push_back(CLS_NUM);

    ma::ann_leaner<DATATYPE> nn(sizes);




    DATATYPE prev_cost = 999.;
    int prev_cost_grow = 0;
    DATATYPE cost_on_save = 999;

    int shift_cnt = 0;

    DATATYPE alpha = .8;

    // for full set
    fill_in_mini_batch(mini_batch_ids, mini_batch_size, train_files_num, buffer, Y.get(), file_id_map, path_train);

    for (int mb = 0; mb < 500000; ++mb) {
        // prepare mini-batch - for not full set
        //fill_in_mini_batch(mini_batch_ids, mini_batch_size, train_files_num, buffer, Y.get(), file_id_map, path_train);

        cout << "# minibatch" << endl;
        for (int i = 1; i < 10; ++i)
            cout << mini_batch_ids[mini_batch_size - i] << ", ";
        cout << endl;



        DATATYPE cost;

        for (int i = 0; i < 2; ++i) {
            cost = nn.fit_minibatch(buffer, Y.get(), mini_batch_size, alpha);
            if (prev_cost < cost)
                alpha /= 2.;
            else if (0.000001 >= ::abs(prev_cost - cost) and alpha < 2.)
                alpha *= 2.;
            else
                prev_cost = cost;
        }


        cout << setprecision(21) << "# last cost: " << cost << " alpha " << alpha << " iter " << mb << endl;

//        if (cost > prev_cost) {
//            cout << "Please adjust alpha: " << endl;
//            cin >> alpha;
//        }



//        nn.print(cout);
//        DATATYPE output[121];
//        nn.get_output(output);
//        cout << "# output vs real " << endl;
//        for (int i = 0; i < 121; ++i)
//            cout << output[i] << "\t" << Y[(mini_batch_size-1)*121 + i] << endl;
//        cout << endl;

//        double v;
//        ma::random::rand(&v, 1);
//        if (.4 <= v) {
//            alpha /= 2.;
//        }
//        else {
//            alpha *= 2.;
//        }



        if (cost <= .30 && cost_on_save > cost) {
            stringstream path;
            path << path_tmp << "ann_100x1_full_" << mb << "_" << cost << ".b";
            fstream fout(path.str().c_str(), fstream::out);
            nn.print(fout);
            fout.close();
            cost_on_save = cost;
            cout << "# saved: " << path.str() << endl;
        }



/*
        shift_cnt += 1;
        ma::random::rand(&v, 1);
        if (5 < prev_cost_grow || (40 < shift_cnt && .7 < v)) {
            prev_cost_grow = 0;
            nn.random_shift();
            shift_cnt = 0;

            ma::random::rand(&v, 1);
            if (.4 <= v) {
                alpha /= 2.;
            }
            else {
                alpha *= 2.;
            }

        }
*/
    }
/*
    {
        int total_bd_size = 0;
        int total_wd_size = 0;

        int layers_num = sizes.size();
        for (int l = 1; l < layers_num; ++l) {
            int l_layer_size = sizes[layers_num - l];

            total_bd_size += l_layer_size;
            total_wd_size += (l_layer_size * sizes[layers_num - l - 1]);
        }


        ma::memory::ptr_vec<DATATYPE> bd, wd;

        nn.calc_deriv(buffer, Y.get(), bd, wd);


        double b_error = 0.;
        double w_error = 0.;

        cout << "BD vs BB_DERIV" << endl;
        for (int b = 0; b < total_bd_size; ++b) {
            cout << bd[b] << "\t" << nn.get_bb_deriv()[b] << endl;
            b_error += ::sqrt((bd[b] - nn.get_bb_deriv()[b]) * (bd[b] - nn.get_bb_deriv()[b]));
        }

        cout << "WD vs WW_DERIV" << endl;
        for (int w = 0; w < total_wd_size; ++w) {
            cout << wd[w] << "\t" << nn.get_ww_deriv()[w] << endl;
            w_error += ::sqrt((wd[w] - nn.get_ww_deriv()[w]) * (wd[w] - nn.get_ww_deriv()[w]));
        }

        cout << "BD avr error: " << (b_error / total_bd_size) << endl;
        cout << "WD avr error: " << (w_error / total_wd_size) << endl;

    }
*/

    return 0;
}



























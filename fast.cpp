

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


typedef float DATATYPE;


const int ALIGN = 512;

const int VEC_LEN = 1 + 2 + 48 * 48;
const int FILE_SIZE_BYTES = 9228;
const int FILE_SIZE = FILE_SIZE_BYTES / sizeof(DATATYPE);
const int CLS_NUM = 121;
//const int CLS_NUM = 2;






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


void load_ids(const char* fname, vector<int>& ids) {
    ids.clear();

    fstream fin(fname, fstream::in);
    string line;
    while (std::getline(fin, line)) {
        int id;
        sscanf(line.c_str(), "%d", &id);
        ids.push_back(id);
    }

    cout << "# ids are initialized " << ids.size() << endl;
}


void load_data(const vector<int>& file_ids, DATATYPE* X, DATATYPE* Y, map<int, pair<string, string> >& file_id_map, const string& path_train) {

    int row = 0;

    // align memory buffer for faster reading
    ma::memory::ptr_vec<DATATYPE> local_buf_non_aligned(new DATATYPE[ ALIGN + FILE_SIZE ]);
    size_t p = (size_t)local_buf_non_aligned.get();
    DATATYPE* local_buf =  (DATATYPE*)((p + ALIGN) - (p % ALIGN));

    for (int i = 0; i < file_ids.size(); ++i) {
        int idx = file_ids[i];
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
        size_t read = fread(local_buf, file_size, 1, fin);

        for (int cls_idx = 0; cls_idx < CLS_NUM; ++cls_idx)
            Y[row * CLS_NUM + cls_idx] = 0.;
        Y[row * CLS_NUM + int(local_buf[0])] = 1.;
        /*
        if (local_buf[0] < 60) {
            Y[row * CLS_NUM + 0] = 1.;
        }
        else {
            Y[row * CLS_NUM + 1] = 1.;
        }
        */

        // this does not include the 1st item which is a class
        int data_vec_len = VEC_LEN - 1;
        for (int val_idx = 0; val_idx < data_vec_len; ++val_idx)
            X[row * data_vec_len + val_idx] = local_buf[1 + val_idx];

        row += 1;

        fclose(fin);
    }

    cout << "# data was loaded" << endl;
}




void save_ann(ma::ann_leaner<DATATYPE>& nn, DATATYPE cost, const char* path_tmp, const char* pref, int iter_num) {
    stringstream path;
    path << path_tmp << pref << "_" << cost << "_" << iter_num << ".b";
    fstream fout(path.str().c_str(), fstream::out);
    nn.print(fout);
    fout.close();
    cout << "# saved: " << path.str() << endl;
}



int main() {

    //
    // Maps ID on file name
    //
    map<int, pair<string, string> > file_id_map;
    get_file_id_map(file_id_map);

    //
    // Some platform specific initialization
    //

    vector<int> train_ids;
    vector<int> valid_ids;

#if defined FOR_LINUX
    const string path_train = "/home/maxim/kaggle/NDSB/data/train_ann1/";
    const char* path_tmp = "/home/maxim/kaggle/NDSB/tmp/";

    load_ids("/home/maxim/kaggle/NDSB/data/train_ids.txt", train_ids);
    load_ids("/home/maxim/kaggle/NDSB/data/valid_ids.txt", valid_ids);
#else
    cout << "# built for Windows" << endl;
    const string path_train = "C:\\Temp\\kaggle\\NDSB\\data\\train_ann1\\";
    const char* path_tmp = "C:\\Temp\\kaggle\\NDSB\\tmp\\";

    load_ids("C:\\Temp\\kaggle\\NDSB\\data\\train_ids.txt", train_ids);
    load_ids("C:\\Temp\\kaggle\\NDSB\\data\\valid_ids.txt", valid_ids);
#endif


    //const int train_files_num = 30336;


    //
    // allocate buffers
    //

    const int vector_len = VEC_LEN - 1; // remove class
    int train_buffer_size = train_ids.size() * vector_len;
    int valid_buffer_size = valid_ids.size() * vector_len;

    ma::memory::ptr_vec<DATATYPE> train_buffer(new DATATYPE[ train_buffer_size ]);
    ma::memory::ptr_vec<DATATYPE> valid_buffer(new DATATYPE[ valid_buffer_size ]);

    ma::memory::ptr_vec<DATATYPE> train_Y (new DATATYPE[train_ids.size() * CLS_NUM]);
    ma::memory::ptr_vec<DATATYPE> valid_Y (new DATATYPE[valid_ids.size() * CLS_NUM]);


    //
    // learn
    //

    ma::random::seed();


    //
    // Prepare ANN
    //

    const char* pref = "ann_10_15";

    vector<int> sizes;
    sizes.push_back(VEC_LEN - 1);   // input
    sizes.push_back(70);
    //sizes.push_back(50);
    sizes.push_back(CLS_NUM);       // output

    ma::ann_leaner<DATATYPE> nn(sizes);




    DATATYPE prev_cost = 999.;
    DATATYPE valid_cost = 0.;
    DATATYPE cost_on_save = 999;
    int small_delta_counter = 0;

    DATATYPE alpha = 1.6;            // learning rate
    DATATYPE lambda = 15;

    //
    // Load data sets
    //
    load_data(train_ids, train_buffer.get(), train_Y.get(), file_id_map, path_train);
    load_data(valid_ids, valid_buffer.get(), valid_Y.get(), file_id_map, path_train);


    //
    // Learning iterations
    //
    for (int mb = 0; mb < 500000; ++mb) {

        {
            fstream fin("./CMD.txt", fstream::in);
            if (fin) {
                string cmd;
                std::getline(fin, cmd);

                if ("ALPHA" == cmd.substr(0, 5)) {
                    DATATYPE a = atof(cmd.substr(6).c_str());
                    alpha = a;
                }
                else if ("SAVE" == cmd) {
                    save_ann(nn, valid_cost, path_tmp, pref, mb);
                }

                //
                fin.close();
                remove("./CMD.txt");
            }
        }

        DATATYPE cost;

        cost = nn.fit_minibatch(train_buffer.get(), train_Y.get(), train_ids.size(), alpha, lambda);
        DATATYPE cost_delta = ::fabs(prev_cost - cost);
        if (prev_cost < cost && .01 <= cost_delta && alpha > .000001)
            alpha /= 2.;

        prev_cost = cost;

        if (alpha < 0.000001) {
            cout << "# local min found, lets try something else, resetting..." << endl;
            save_ann(nn, valid_cost, path_tmp, pref, mb);

            nn.random_shift();
            alpha = 1.6;
            prev_cost = 999.;
            cost_on_save = 999.;
            mb = 0;
        }


        if (0 == (mb % 5)) {
            valid_cost = 0.;
            for (int vr = 0; vr < valid_ids.size(); ++vr) {
                nn.forward(&valid_buffer[vr * vector_len]);
                valid_cost += nn.cost(&valid_Y[vr * CLS_NUM]);
            }
            valid_cost /= valid_ids.size();
            cout << setprecision(21) << "# cost: " << cost << " (" << cost_delta << ") alpha " << alpha << " iter " << mb << " validation " << valid_cost << endl;
        }
        else {
            cout << setprecision(21) << "# cost: " << cost << " (" << cost_delta << ") alpha " << alpha << " iter " << mb << endl;
        }

        if (cost_delta < 0.00019) {
            ++small_delta_counter;
            if (small_delta_counter > 10) {
                small_delta_counter = 0;
                alpha *= 2.;
            }
        }


        if (0 == ((mb+1) % 100) || cost <= .30 && cost_on_save > valid_cost) {
            save_ann(nn, valid_cost, path_tmp, pref, mb);
            cost_on_save = valid_cost;
        }
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



























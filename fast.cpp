

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>

#include <random.hpp>
#include <ann.hpp>
#include <id_file_map.h>



using namespace std;


typedef double DATATYPE;


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


void fill_in_mini_batch(int* mini_batch_ids, int mini_batch_size, int train_files_num, float* buffer, map<int, pair<string, string> >& file_id_map, const string& path_train) {

    ma::random::get_k_of_n(mini_batch_size, train_files_num, mini_batch_ids);

    int idx = 0;

    for (int i = 0; i < mini_batch_size; ++i) {
        pair<string, string>& p = file_id_map[ mini_batch_ids[i] ];
        string fname = path_train + p.first + "\\" + p.second + ".b";

        FILE* fin = fopen(fname.c_str(), "rb");

        // get file size
        fseek(fin, 0, SEEK_END);
        size_t file_size = ftell(fin) / sizeof(float);  // in floats
        fseek(fin, 0, SEEK_SET);

        // read
        size_t read = fread(&buffer[idx], file_size * sizeof(float), 1, fin);
        idx += file_size;

        //cout << "# loaded [" << mini_batch_ids[i] << "]: " << fname << " (" << file_size << ")" << endl;

        fclose(fin);
    }
}




int main() {

    map<int, pair<string, string> > file_id_map;
    get_file_id_map(file_id_map);

    const string path_train = "C:\\Temp\\kaggle\\NDSB\\data\\train_ann1\\";

    const int train_files_num = 30336;

    const int mini_batch_size = 2500;
    int mini_batch_ids[mini_batch_size];

    // allocate mini-batch buffer
    const int vector_len = 1 + 100 * 100;
    ma::memory::ptr_vec<float> buffer(new float[ mini_batch_size * vector_len ]);


    // learn
    ma::random::seed();


    vector<int> sizes;
    sizes.push_back(10000);
    sizes.push_back(15);
    sizes.push_back(121);

    ma::ann_leaner<DATATYPE> nn(sizes);

    DATATYPE y[121];

    DATATYPE new_x[10000];

    for (int mb = 0; mb < 10000; ++mb) {
        // prepare mini-batch
        fill_in_mini_batch(mini_batch_ids, mini_batch_size, train_files_num, buffer.get(), file_id_map, path_train);


        DATATYPE cost = 999999.;
        DATATYPE prev_cost = cost;

        DATATYPE alpha = 10.; //5.;


        for (int v = 0; v < mini_batch_size; ++v) {
            int idx = v * vector_len;

            for (int i = 0; i < 121; ++i)
                y[i] = 0;
            y[ int(buffer[idx + 0]) ] = 1.;

            for (int i = 0; i < 10000; ++i)
                new_x[i] = buffer[idx + 1 + i];

            //cost = nn.fit_minibatch(&buffer[idx + 1], y, 1, alpha);
            cost = nn.fit_minibatch(&new_x[0], y, 1, alpha);

            if (prev_cost < cost) {
                alpha /= 2.;
            }
            prev_cost = cost;
        }
        cout << "# last cost: " << cost << endl;

        nn.print(cout);
    }
/*
    {
        int total_bd_size = 136;
        int total_wd_size = 151815;

        ma::memory::ptr_vec<DATATYPE> bd, wd;

        for (int i = 0; i < 121; ++i)
            y[i] = 0;
        y[ int(buffer[0]) ] = 1.;

        nn.calc_deriv(&new_x[0], y, bd, wd);

        cout << "BD vs BB_DERIV" << endl;
        for (int b = 0; b < total_bd_size; ++b)
            cout << bd[b] << "\t" << nn.get_bb_deriv()[b] << endl;

        cout << "WD vs WW_DERIV" << endl;
        for (int w = 0; w < total_wd_size; ++w)
            cout << wd[w] << "\t" << nn.get_ww_deriv()[w] << endl;

    }
*/

    return 0;
}




























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
    }
}




int main() {

    map<int, pair<string, string> > file_id_map;
    get_file_id_map(file_id_map);

    const string path_train = "C:\\Temp\\kaggle\\NDSB\\data\\train_ann1\\";

    const int train_files_num = 30336;

    const int mini_batch_size = 10000;
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

    ma::ann_leaner<float> nn(sizes);

    float y[121];

    for (int mb = 0; mb < 10; ++mb) {
        // prepare mini-batch
        fill_in_mini_batch(mini_batch_ids, mini_batch_size, train_files_num, buffer.get(), file_id_map, path_train);


        float cost = 999999.;
        float prev_cost = cost;

        float alpha = 5.;

        for (int v = 0; v < mini_batch_size; ++v) {
            int idx = v * vector_len;

            memset(y, 0, 121 * sizeof(float));
            y[ int(buffer[idx + 0]) ] = 1.;

            cost = nn.fit_minibatch(&buffer[idx + 1], y, 1, alpha);
            if (prev_cost < cost) {
                alpha /= 2.;
            }
            prev_cost = cost;
        }
        cout << "# last cost: " << cost << endl;

        nn.print(cout);
    }


    return 0;
}


























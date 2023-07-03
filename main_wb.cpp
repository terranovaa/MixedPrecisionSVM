#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include "external/cppposit_private/include/posit.h" // include non-tabulated posits

using real = posit::Posit<int8_t, 8, 0, uint8_t, posit::PositSpec::WithNan>;

template <typename T>
struct svm_parameters {
    std::vector<T> w;
    T b;
};

template <typename T>
T dot_product(const std::vector<T> x1, const std::vector<T> x2){
    T res = T(0);
    if (x1.size() != x2.size()){
        std::cerr << "Inner product requires the same number of elements." << std::endl;
        exit(-1);
    }
    for(int i = 0; i < x1.size(); i++){
        res += x1[i] * x2[i];
    }
    return res;
}

template <typename T>
void read_dataset(std::string filename, std::vector<std::vector<T>> &ret) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Error opening file" << std::endl;
        return;
    }
    std::string line;
    while (getline(file, line)) {
        std::vector<T> row;
        std::stringstream ss(line);
        std::string cell;
        double tmp;
        while (getline(ss, cell, ',')) {
            try {
                std::stringstream ss1(cell);
                ss1 >> tmp;
                row.push_back(T(tmp));
            } catch (const std::exception& e) {
                std::cout << "Error: " << e.what() << std::endl;
                return;
            }
        }
        ret.push_back(row);
    }
    file.close();
}

// Function used to perform inference with a single sample
template <typename T>
T predict_point(std::vector<T> x, std::vector<T> w, T b){
    T threshold = T(0);
    T result = dot_product(x,w) + b;
    if(result >= threshold){
        return T(+1);
    } else {
        return T(-1);
    }
}

// Function used to perform inference with the overall dataset
template <typename T>
std::vector<T> predict_dataset(std::vector<std::vector<T>> x, std::vector<T> w, T b){
    T threshold = T(0);
    std::vector<T> classes;
    for (int i = 0; i < x.size(); i++)
        classes.push_back(predict_point(x[i], w, T(b)));
    return classes;
}

// Function used to compute accuracy
template <typename T>
float model_evaluate(std::vector<T> p, std::vector<T> y){
    float res = 0;
    if (p.size() != y.size()){
        std::cerr << "Inner product requires the same number of elements." << std::endl;
        exit(-1);
    }
    for(int i = 0; i < p.size(); i++){
        res += (p[i] == y[i]);
    }
    return (float)res/p.size();
}

// Function used to read the weight vector
template <typename T>
void get_w(std::string filename, std::vector<T> &ret) {
    std::string line;
    std::ifstream file(filename);
    if (file.is_open()) {
        while (std::getline(file, line)) {
            double tmp;
            std::stringstream ss(line);
            std::getline(ss, line);
            std::stringstream ss1(line);
            ss1 >> tmp;
            T w(tmp);
            ret.push_back(w);
        }
        file.close();
    }
    else {
        std::cout << "Unable to open file" << std::endl;
    }
}

template <typename T>
void test_wb_precomputed(std::vector<std::vector<T>> X,std::vector<T> y, std::string weightVectorFile, float b){
    // represent w and b as posits
    svm_parameters<T> opt;
    opt.b = b;
    std::cout << "Scalar b: "<< std::endl;
    std::cout << opt.b << std::endl;
    get_w(weightVectorFile, opt.w);
    std::cout << "Vector w: ";
    for (int i = 0; i < opt.w.size()-1; i++) {
        std::cout << opt.w[i] << ",";
    }
    std::cout << opt.w[opt.w.size()] << std::endl;
    // perform inference on dataset
    std::vector<T> p = predict_dataset(X,opt.w,opt.b);
    // evaluate accuracy
    float acc = model_evaluate(p, y);
    std::cout << "Accuracy: " << acc << std::endl;
}


int main() {
    // sets the dataset file (test set) to classify
    std::string dataset = "./datasets/banknote_test_normalized.csv";
    // set the file containing the weight vector
    std::string weightVectorFile = "./matlab/w_BN_81.csv";
    // set the optimal value of the scalar b
    float b = -1.28494823385437;

    // reads the dataset
    std::vector<std::vector<double>> dataList;
    read_dataset(dataset, dataList);
    if(dataList.empty())
        std::cout << "Error reading the dataset" << std::endl;

    std::vector<std::vector<real>> X;
    std::vector<real> y;
    real elem;
    for (int i = 0; i < dataList.size(); i++) {
        std::vector<real> row;
        for (int j = 0; j < dataList[i].size()-1; j++) { // exclude the label from X
            elem = dataList[i][j];
            row.push_back(elem);
        }
        X.push_back(row);
    }

    for (int i = 0; i < dataList.size(); i++) {
        elem = dataList[i][dataList[i].size()-1]; // save in y just the label
        y.push_back(elem);
    }

    // Print the overall dataset and label
    /*
    std::cout << "Dataset: " << std::endl;
    for (int i = 0; i < X.size(); i++) {
        int j;
        for (j = 0; j < X[i].size()-1; j++) {
            std::cout << X[i][j] << ",";
        }
        std::cout << X[i][j] << "," << y[i] << std::endl;
    }
    */

    // decision function computation on dataset
    test_wb_precomputed(X,y,weightVectorFile,b);

    return 0;
}


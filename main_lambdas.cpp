#include <iostream>
#include <vector>
#include <algorithm> 
#include <fstream>
#include <cmath>
#include "external/cppposit_private/include/posit.h" // include non-tabulated posits

using real = posit::Posit<int8_t, 8, 0, uint32_t, posit::PositSpec::WithInfs>;
using real1 = posit::Posit<int16_t, 16, 2, uint32_t, posit::PositSpec::WithInfs>;

// real numbers error tolerance
const double epsilon = 1e-5;
// parameter of the optimization problem
double C = 1;

template <typename T>
struct svm_parameters {
    std::vector<T> w;
    T b;
};

// Function used to read lambdas from a given file
template <typename T>
void get_lambdas(std::string &filename, std::vector<T> &ret) {
    std::string line;
    std::ifstream file(filename);
    if (file.is_open()) {
        while (std::getline(file, line)) {
            double tmp;
            std::stringstream ss(line);
            std::getline(ss, line,',');
            std::stringstream ss1(line);
            ss1 >> tmp;
            T lambda(tmp);
            ret.push_back(lambda);
        }
        file.close();
    }
    else {
        std::cout << "Unable to open file" << std::endl;
    }
}

// Function used to read variables mu and eta from a given file and compute their difference to obtain lambdas
template <typename T>
void get_lambdas_from_mu_eta(std::string &filename, std::vector<T> &ret) {
    std::string line;
    std::ifstream file(filename);
    if (file.is_open()) {
        while (std::getline(file, line)) {
            double tmp;
            std::stringstream ss(line);
            // Read mu
            std::getline(ss, line,',');
            std::stringstream ss1(line);
            ss1 >> tmp;
            T mu(tmp);
            // Read eta
            std::getline(ss, line,',');
            std::stringstream ss2(line);
            ss2 >> tmp;
            T eta(tmp);
            T diff = mu - eta; // lambda
            ret.push_back(diff);
        }
        file.close();
    }
    else {
        std::cout << "Unable to open file" << std::endl;
    }
}

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

// Function used to compute the weight vector and the scalar b with a smart approach using posits
// and exploiting as much as possible the dense region of the posit ring
template <typename T>
svm_parameters<T> get_smart_SVM_parameters(std::vector<T> lambdas, std::vector<std::vector<T>> x, std::vector<T> y) {
    svm_parameters<T> opt;
    // compute partial products to be summed, for each component of the vector w
    std::vector<std::vector<T>> wi;
    for (int j = 0; j < x[0].size(); ++j) {
        std::vector<T> internal_wi;
        for (int i = 0; i < x.size(); ++i) {
            internal_wi.push_back(lambdas[i] * y[i] * x[i][j]);
        }
        wi.push_back(internal_wi);
    }
    // select an upper bound
    T UB = 1.0;
    std::vector<T> w;
    for(int i = 0; i < wi.size(); ++i){
        std::cout << "wi computation with index" << i << std::endl;
        T value = 0;
        // start from the left contribution (hopefully negative)
        bool left = true;
        // order the contributions of each component of the vector
        std::sort(wi[i].begin(), wi[i].end());
        int leftIndex = 0;
        int rightIndex = wi[i].size();
        int tmp;
        while(true){
            if(leftIndex == rightIndex)
                break;
            value += wi[i][leftIndex];
            // swap to the right (hopefully positive)
            if(value <= -UB && left){
                // the condition left == true is needed otherwise we swap conditions repeatedly if the upper bound
                // is not restored at first time after swapping
                std::cout << std::endl <<  "Exchanging order to positive because of value " << value << std::endl;
                tmp = leftIndex;
                leftIndex = rightIndex;
                rightIndex = tmp;
                left = false;
            } else if(value >= UB && !left){ // swap to the left (hopefully negative)
                std::cout << std::endl << "Exchanging order to negative because of value " << value << std::endl;
                tmp = leftIndex;
                leftIndex = rightIndex;
                rightIndex = tmp;
                left = true;
            }
            // increase index accordingly to the position
            if(left)
                leftIndex++;
            else
                leftIndex--;
        }
        std::cout << std::endl;
        w.push_back(value);
    }
    T b;
    for (int i = 0; i < x.size(); i++) {
        // find first i to compute b
        if (lambdas[i] >= T(epsilon) && lambdas[i] <= T(C-epsilon)) {
            b = y[i] - dot_product(w, x[i]);
            break;
        }
    }
    opt.b = T(b);
    opt.w = w;
    return opt;
}

// Function used to compute the weight vector and the scalar b using the classic approach and posits
template <typename T>
svm_parameters<T> get_SVM_parameters(std::vector<T> lambdas, std::vector<std::vector<T>> x, std::vector<T> y) {
    svm_parameters<T> opt;
    for (int j = 0; j < x[0].size(); ++j) {
        T w = 0;
        for (int i = 0; i < x.size(); ++i)
            w += lambdas[i] * y[i] * x[i][j];
        opt.w.push_back((w));
    }

    T b;
    for (int i = 0; i < x.size(); i++) {
        if (lambdas[i] >= T(epsilon) && lambdas[i] <= T(C-epsilon)) {
            b = y[i] - dot_product(opt.w, x[i]);
            break;
        }
    }
    opt.b = b;
    return opt;
}

// Function used to obtain the weight vector and the scalar b using the dual variables and exploiting higher precision types
template <typename T>
svm_parameters<T> get_SVM_parameters_with_high_precision(std::vector<T> lambdas, std::vector<std::vector<T>> x, std::vector<T> y) {
    svm_parameters<T> opt;
    std::vector<real1> _w;
    for (int j = 0; j < x[0].size(); ++j) {
        real1 w = 0;
        for (int i = 0; i < x.size(); ++i)
            w += real1((double)lambdas[i]) * real1((double)y[i]) * real1((double)x[i][j]);
        _w.push_back(w);
        opt.w.push_back(T((double)w));
    }

    std::vector<std::vector<real1>> _x;
    for (auto row : x) {
        std::vector<real1> _row;
        for (auto cell : row)
            _row.push_back((double)cell);
        _x.push_back(_row);
    }

    real1 b;
    for (int i = 0; i < x.size(); i++) {
        if (lambdas[i] >= T(epsilon) && lambdas[i] <= T(C-epsilon)) {
            b = real1((double)y[i]) - dot_product(_w, _x[i]);
            break;
        }
    }

    opt.b = T((double)b);
    return opt;
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

// Function used to implement a gaussian kernel
template <typename T>
T kernel(std::vector<T> x, std::vector<T> xi, float gamma){
    std::vector<T> diff (x.size());
    for(int i = 0; i < x.size(); ++i){
        diff[i] = (x[i] - xi[i]);
    }
    T norm = sqrt(dot_product(diff,diff));
    return exp(-gamma * norm * norm);
}


// Function used to perform inference with a single sample and non-linear kernel exploiting higher precision types
template <typename T>
T predict_non_linear_point(std::vector<std::vector<T>> x, std::vector<T> y, std::vector<T> lambdas, std::vector<T> x_test) {
    int index = -1;
    float epsilonDouble = 1e-3;
    std::vector<std::vector<float>> xHighP (x.size());
    for(int i = 0; i < x.size(); ++i){
        xHighP[i] = std::vector<float>(x[i].size());
        for(int j = 0; j < x[i].size(); ++j){
            xHighP[i][j] = x[i][j];
        }
    }
    std::vector<float> yHighP (y.size());
    for(int i = 0; i < y.size(); ++i){
        yHighP[i] = y[i];
    }
    std::vector<float> lambdaHighP (lambdas.size());
    for(int i = 0; i < lambdas.size(); ++i){
        lambdaHighP[i] = lambdas[i];
    }
    std::vector<float> xTestHighP (x_test.size());
    for(int i = 0; i < x_test.size(); ++i){
        xTestHighP[i] = x_test[i];
    }
    float b = -1;
    for(int i = 0; i < xHighP.size(); ++i){
       if (lambdaHighP[i] >= epsilonDouble && lambdaHighP[i] <= C-epsilonDouble) {
           b = 1/yHighP[i];
           index = i;
           break;
       }
    }
    for(int i = 0; i < xHighP.size(); ++i){
        b = b - lambdaHighP[i]*yHighP[i]*kernel(xHighP[index], xHighP[i], 0.001);
    }
    double resultHighP = 0;
    for (int i = 0; i < xHighP.size(); ++i) {
        resultHighP = resultHighP + lambdaHighP[i] * yHighP[i] * kernel(xHighP[i], xTestHighP, 0.001);
    }
    resultHighP = resultHighP + b;
    T result = resultHighP;
    T threshold = T(0);
    if(result >= threshold){
        return +1;
    } else {
        return -1;
    }
}

// Function used to perform inference with the overall dataset
template <typename T, typename T1>
std::vector<T1> predict_dataset(std::vector<std::vector<T1>> x, std::vector<T> w, T b){
    std::vector<T1> _w (w.size());
    for (int i=0; i<w.size(); i++)
        _w[i] = T1((double) w[i]);

    T threshold = T(0);
    std::vector<T1> classes;
    for (int i = 0; i < x.size(); i++)
        classes.push_back(predict_point(x[i], _w, T1((double) b)));
    return classes;
}

// Function used to perform inference with the overall dataset in case of non-linear kernels
template <typename T>
std::vector<T> predict_non_linear_dataset(std::vector<std::vector<T>> x_test, std::vector<std::vector<T>> x, std::vector<T> y, std::vector<T> lambdas){
    std::vector<T> classes;
    for (int i = 0; i < x_test.size(); i++)
        classes.push_back(predict_non_linear_point(x, y, lambdas, x_test[i]));
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

int main() {
    std::string filename = "./matlab/mueta.csv";
    // should be a file containing mus and etas or lambdas, and the corresponding function should be properly chosen below
    std::string train_data_filename = "./datasets/breast_cancer_train_normalized.csv";
    std::string test_data_filename = "./datasets/breast_cancer_test_normalized.csv";
    bool linearVersion = true;

    std::vector<real> lambdas;

    // Compute the support vectors from the mu and eta stored in the file
    get_lambdas_from_mu_eta(filename, lambdas);

    // Get lambdas from the file
    //get_lambdas(filename, lambdas);

    // reads the training and test sets
    std::vector<std::vector<double>> training_set, test_set;
    read_dataset(train_data_filename, training_set);
    read_dataset(test_data_filename, test_set);

    if(training_set.empty() || test_set.empty()) {
        std::cout << "Error reading the dataset" << std::endl;
        return -1;
    }

    // training set
    std::vector<std::vector<real>> X_train;
    std::vector<real> y_train;
    real elem;
    for (int i = 0; i < training_set.size(); i++) {
        std::vector<real> row;
        for (int j = 0; j < training_set[i].size()-1; j++) {  // exclude the label
            elem = training_set[i][j];
            row.push_back(elem);
        }
        X_train.push_back(row);
    }

    for (int i = 0; i < training_set.size(); i++) {
        elem = training_set[i][training_set[i].size()-1]; // save in y just the label
        y_train.push_back(elem);
    }

    //  Test set
    std::vector<std::vector<real>> X_test;
    std::vector<real> y_test;

    for (int i = 0; i < test_set.size(); i++) {
        std::vector<real> row;
        for (int j = 0; j < test_set[i].size()-1; j++) { // exclude the label
            elem = test_set[i][j];
            row.push_back(elem);
        }
        X_test.push_back(row);
    }

    for (int i = 0; i < test_set.size(); i++) {
        elem = test_set[i][test_set[i].size()-1]; // save in y just the label
        y_test.push_back(elem);
    }

    // Print the overall dataset and label
    /*
    std::cout << "Dataset to predict (test set): " << std::endl;
    for (int i = 0; i < X_test.size(); i++) {
        int j;
        for (j = 0; j < X_test[i].size()-1; j++) {
            std::cout << X_test[i][j] << ",";
        }
        std::cout << X_test[i][j] << "," << y_test[i] << std::endl;
    }
    */

    if(linearVersion) {
        svm_parameters<real> opt = get_SVM_parameters(lambdas, X_train, y_train);
        // Uncomment for the higher precision type version:
        // svm_parameters<real> opt = get_SVM_parameters_with_high_precision(lambdas, X_train, y_train);
        // Uncomment for the smart version:
        // svm_parameters<real> opt = get_smart_SVM_parameters(lambdas, X_train, y_train);

        std::cout << "Vector w: ";
        for (int i = 0; i < opt.w.size() - 1; i++) {
            std::cout << opt.w[i] << ",";
        }
        std::cout << opt.w[opt.w.size()] << std::endl;
        std::cout << "Scalar b: " << opt.b << std::endl;

        std::vector<real> p = predict_dataset(X_test, opt.w, opt.b);

        float acc = model_evaluate(p, y_test);
        std::cout << "Accuracy: " << acc << std::endl;
    } else {
        std::vector<real> p = predict_non_linear_dataset(X_test, X_train, y_train, lambdas);
        float acc = model_evaluate(p, y_test);
        std::cout << "Accuracy: " << acc << std::endl;
    }
    return 0;
}
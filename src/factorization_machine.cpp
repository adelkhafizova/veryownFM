//
// Created by Адель Хафизова on 13.05.18.
//
#include "factorization_machine.h"
#include <sstream>
#include <iostream>
#include <string>


FactorizationMachine::FactorizationMachine(float lr, const std::string &reg_const, int num_iter,
                                           const TaskType &type, bool use_bias, bool use_linear, int pairwise_rank,
                                           int max_feature, float max_target, float min_target) {
    _learning_rate = lr;
    if (reg_const.size()) {
        std::size_t found = reg_const.find(',');
        if (found != std::string::npos) {
            const char *pline = reg_const.c_str();
            int shift;
            sscanf(pline, "%lf,%n", &_reg_w0, &shift);
            pline += shift;
            sscanf(pline, "%lf,%n", &_reg_w, &shift);
            pline += shift;
            sscanf(pline, "%lf,%n", &_reg_v, &shift);
        } else {
            const char *pline = reg_const.c_str();
            sscanf(pline, "%lf", &_reg_w0);
            _reg_w = _reg_w0;
            _reg_v = _reg_w0;
        }
    }
    _iterations = num_iter;
    _task_type = type;
    _max_feature = max_feature;
    _max_target = max_target;
    _min_target = min_target;
    _k_0 = use_bias;
    _k_1 = use_linear;
    _k_2 = pairwise_rank;
    _w = std::vector<double>(_max_feature + 1, 0.0);
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0,0.1);
    _v = std::vector<std::vector<double> >(_max_feature + 1, std::vector<double>(_k_2, 0.0));
    for (int i = 0; i < _max_feature + 1; ++i) {
        for (int j = 0; j < _k_2; ++j) {
            _v.at(i).at(j) = distribution(generator);
        }
    }
}

void FactorizationMachine::launch_learning(Dataset *train_data, Dataset *test_data) {
    for (int i = 0; i < _iterations; i++) {
        learn_step(train_data);
        std::cout << "iter=" << i << " ";
        std::cout << "Train=" << evaluate(train_data) << " ";
        std::cout << "Test=" << evaluate(test_data) << std::endl;
    }
}

double FactorizationMachine::evaluate(Dataset *data) {
    double error = 0.0;
    for (int i = 0; i < data->size(); ++i) {
        float y_hat = predict(data);
        y_hat = std::max(y_hat, _min_target);
        y_hat = std::min(y_hat, _max_target);
        double target = (double) data->get_target();
        if (_task_type == classification) {
            //TODO: check that this function is the correct way to measure
            error += -target * log(y_hat + 1e-9) - (1 - target) * log(1 - y_hat + 1e-9);
        } else {
            error += pow(y_hat - target, 2);
        }
        data->next_row();
    }
    if (_task_type == classification) {
        return error / data->size();
    } else {
        return sqrt(error/data->size());
    }
}

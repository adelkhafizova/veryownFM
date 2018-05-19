//
// Created by Адель Хафизова on 13.05.18.
//
#include "fm_sgd.h"

FactorizationMachineSGD::FactorizationMachineSGD(float lr, const std::string &reg_const, int num_iter,
                                                 const TaskType &type, bool use_bias, bool use_linear, int pairwise_rank,
                                                 int max_feature, float max_target, float min_target):
    FactorizationMachine(lr, reg_const, num_iter, type, use_bias, use_linear, pairwise_rank, max_feature, max_target, min_target) {
    _linear_vx_sum = std::vector<double> (_k_2, 0.0);
}

void FactorizationMachineSGD::learn_step(Dataset *train_data) {
    for (int i = 0; i < train_data->size(); ++i) {
        double y_hat = predict(train_data);
        double target = (double)train_data->get_target();
        auto row = train_data->get_row();
        double coefficient = 0;
        if (_task_type == classification) {
            coefficient = (1 / (1 + exp(-target * y_hat)) - 1) * target;
        } else {
            coefficient = (y_hat - target);
        }
        sgd_step(row, coefficient);
        train_data->next_row();
    }
}

double FactorizationMachineSGD::predict(Dataset const *data) {
    double y_hat = 0.0;
    const std::map<int, float> row = data->get_row();

    for (int f = 0; f < _k_2; f++) {
        _linear_vx_sum.at(f) = 0.0;
        double sum_of_squared = 0.0;
        for (auto it = row.begin(); it != row.end(); it++) {
            double sum_element = _v.at(it->first).at(f)*it->second;
            _linear_vx_sum.at(f) += sum_element;
            sum_of_squared += sum_element*sum_element;
        }
        y_hat += _linear_vx_sum.at(f)*_linear_vx_sum.at(f) - sum_of_squared;
    }
    y_hat *= 0.5;

    if (_k_0) {
        y_hat += _w_0;
    }


    if (_k_1) {
        for (auto it = row.begin(); it != row.end(); it++) {
            y_hat += _w.at(it->first)*it->second;
        }
    }
    if (_task_type == classification) {
        y_hat = 1.0 / (1.0 + exp(-y_hat));
    }
    return y_hat;
}

void FactorizationMachineSGD::sgd_step(const std::map<int, float> &row, double coefficient) {
    for (int f = 0; f < _k_2; f++) {
        for (auto it = row.begin(); it != row.end(); it++) {
            double grad = coefficient;
            grad *= it->second*(_linear_vx_sum.at(f) - _v.at(it->first).at(f)*it->second);
            grad = std::min(grad, 5.);
            grad = std::max(grad, -5.);
            _v.at(it->first).at(f) -= _learning_rate*(grad + 2*_reg_v*_v.at(it->first).at(f));
        }
    }

    if (_k_0) {
        double grad = coefficient;
        _w_0 -= _learning_rate*(grad + 2*_reg_w0);
    }
    if (_k_1) {
        for (auto it = row.begin(); it != row.end(); it++) {
            double grad = coefficient;
            grad *= it->second;
            _w.at(it->first) -= _learning_rate*(grad + 2*_reg_w*_w.at(it->first));
        }
    }
}

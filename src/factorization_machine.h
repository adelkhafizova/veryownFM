//
// Created by Адель Хафизова on 26.04.18.
//

#ifndef FM_FACTORIZATION_MACHINE_H
#define FM_FACTORIZATION_MACHINE_H

#endif //FM_FACTORIZATION_MACHINE_H

class FactorizationMachine {
public:
    FactorizationMachine(const float &lr, const float &reg_const, int &num_iter, learning_method &lm, task_type &type,
                         const int & max_feature, const float & max_target, const float & min_target) {
        _learning_rate = lr;
        _regularization_const = reg_const;
        _iterations = num_iter;
        _learning_method = lm;
        _task_type = type;
        _max_feature = max_feature;
        _max_target = max_target;
        _min_target = min_target;
        std::cout << _task_type << std::endl;
        _w = std::vector<double>(_max_feature + 1, 0.0);
        _v = std::vector<std::vector<double> >(_max_feature + 1, std::vector<double>(_k_2, 0.0));
        _linear_vx_sum = std::vector<double> (_k_2, 0.0);
    };

    void learn_step(const Dataset &train_data, const Dataset &test_data, int iteration) {
        if (_learning_method == SGD) {
            for (int i = 0; i < train_data.size(); ++i) {
                double y_hat = predict(train_data, i);
                //std::cout << y_hat << std::endl;
                double target = (double)train_data.get_target(i);
                auto row = train_data.get_row(i);
                double coefficient = 0.0;
                if (_task_type == classification) {
                    coefficient = (1 / (1 + exp(target * y_hat)) - 1) * target;
                } else {
                    coefficient = 2 * (y_hat - target);
                }
                sgd_step(row, coefficient);
            }
            std::cout << iteration << " " << evaluate(train_data) << std::endl;
            std::cout << iteration << " " << evaluate(test_data) << std::endl;
            return;
        }
        if (_learning_method == ALS) {
            return;
        }
    }

    double predict(const Dataset &train_data, int i) {
        float y_hat = 0.0;
        const std::map<int, float> row = train_data.get_row(i);

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

        } else {
            y_hat = std::max(y_hat, _min_target);
            y_hat = std::min(y_hat, _max_target);
        }

        return y_hat;
    }

    void sgd_step(const std::map<int, float> &row, double coefficient) {
        for (int f = 0; f < _k_2; f++) {
            for (auto it = row.begin(); it != row.end(); it++) {
                double grad = coefficient;
                grad *= it->second*(_linear_vx_sum.at(f) - _v.at(it->first).at(f)*it->second);
                _v.at(it->first).at(f) -= _learning_rate*grad;
            }
        }

        if (_k_0) {
            double grad = coefficient;
            _w_0 -= _learning_rate*grad; //TODO regularization
        }
        if (_k_1) {
            for (auto it = row.begin(); it != row.end(); it++) {
                double grad = coefficient;
                grad *= it->second;
                _w.at(it->first) -= _learning_rate*grad;
            }
        }
    }

    double evaluate(const Dataset &data) {
        double error = 0.0;
        for (int i = 0; i < data.size(); ++i) {
            double y_hat = predict(data, i);
            double target = (double) data.get_target(i);
            if (_task_type == classification) {
                //TODO: check that this function is the correct way to measure
                error += -target*log(y_hat) - (1 - target)*log(1 - y_hat);
            } else {
                error += pow(y_hat - target, 2);
            }
        }
        if (_task_type == classification) {
            return error/data.size();
        } else {
            return sqrt(error/data.size());
        }
    }
    void launch_learning(const Dataset & train_data, const Dataset & test_data) {
        for (int i = 0; i < _iterations; i++) {
            learn_step(train_data, test_data, i);
        }
    }

private:
    float _learning_rate;
    float _regularization_const;
    int _iterations;
    learning_method _learning_method;
    task_type _task_type;
    bool _k_0 = true;
    bool _k_1 = true;
    int _k_2 = 8;
    double _w_0 = 0.0;
    int _max_feature;
    float _min_target;
    float _max_target;
    double _reg_w0 = 0.0;
    double _reg_w = 0.0;
    double _reg_v = 0.0;
    //int _max_target;
    //int _min_target;
    std::vector<double> _w;
    std::vector<std::vector<double> > _v;
    std::vector<double> _linear_vx_sum;
};
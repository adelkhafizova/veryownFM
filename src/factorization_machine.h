//
// Created by Адель Хафизова on 26.04.18.
//

#ifndef FM_FACTORIZATION_MACHINE_H
#define FM_FACTORIZATION_MACHINE_H

#endif //FM_FACTORIZATION_MACHINE_H
#include <random>


class FactorizationMachine {
public:
    FactorizationMachine(float lr, const std::string &reg_const, int num_iter, const learning_method &lm,
                         const task_type &type, int max_feature, float max_target, float min_target) {
        _learning_rate = lr;
        /*if (reg_const.size()) {
            if
            const char *pline = reg_const.c_str();

        }*/
        _iterations = num_iter;
        _learning_method = lm;
        _task_type = type;
        _max_feature = max_feature;
        _max_target = max_target;
        _min_target = min_target;
        _w = std::vector<double>(_max_feature + 1, 0.0);
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0.0,0.1);
        _v = std::vector<std::vector<double> >(_max_feature + 1, std::vector<double>(_k_2, 0.0));
        for (int i = 1; i < _max_feature + 1; ++i) {
            for (int j = 0; j < _k_2; ++j) {
                _v.at(i).at(j) = distribution(generator);
            }
        }
        _linear_vx_sum = std::vector<double> (_k_2, 0.0);
    };

    void learn_step(const Dataset &train_data, const Dataset &test_data, int iteration) {
        if (_learning_method == SGD) {
            for (int i = 0; i < train_data.size(); ++i) {
                double y_hat = predict(train_data, i);
                double target = (double)train_data.get_target(i);
                auto row = train_data.get_row(i);
                double coefficient = 0;
                if (_task_type == classification) {
                    coefficient = (1 / (1 + exp(target * y_hat)) - 1) * target;
                } else {
                    coefficient = (y_hat - target);
                }
                sgd_step(row, coefficient);
            }
        } else {
            double _errors_sum = 0.0;
            double _x_square = 0.0;
            for (int i = 0; i < train_data.size(); ++i) {
                double y_hat = predict(train_data, i);
                _errors.at(i) = (double)train_data.get_target(i) - y_hat;
                _errors_sum += _errors.at(i);
                auto row = train_data.get_row(i);

                for (int f = 0; f < _k_2; f++) {
                    _cache.at(i).at(f) = 0.0;
                    for (auto it = row.begin(); it != row.end(); it++) {
                        _cache.at(i).at(f) += _v.at(it->first).at(f)*it->second;
                    }
                }
            }
            als_step(train_data, _errors_sum);
        }
        std::cout << "iter=" << iteration << " ";
        std::cout << "Train=" << evaluate(train_data) << " ";
        std::cout << "Test=" << evaluate(test_data) << std::endl;
        return;
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
            /*if (_learning_method == ALS && first_traverse) {
                _cache.at(i).at(f) = _linear_vx_sum.at(f);
            }*/
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
                grad = std::min(grad, 10.0);
                grad = std::max(grad, -10.0);
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

    void als_step(const Dataset &train_data, double _errors_sum) {
        double delta = -_w_0;
        _w_0 = (_w_0 * train_data.size() + _errors_sum) / (train_data.size() + _reg_w0);
        delta += _w_0;
        for (int i = 0; i < train_data.size(); ++i) {
            _errors.at(i) += delta;
        }
        for (int l = 0; l < _max_feature + 1; ++l) {
            double delta = -_w.at(l);
            double _w_l_star = 0.0;
            double x_l_square = 0.0;
            const std::vector <int> &valid_objects = train_data.get_feature_objects(l);
            for (auto it = valid_objects.begin(); it != valid_objects.end(); ++it) {
                const std::map<int, float> row = train_data.get_row(*it);
                auto element = row.find(l);
                _w_l_star += element->second * (_w.at(l) * element->second - _errors.at(*it));
                x_l_square += element->second*element->second;
            }
            _w_l_star /= x_l_square + _reg_w;
            delta += _w_l_star;
            _w.at(l) = _w_l_star;
            for (auto it = valid_objects.begin(); it != valid_objects.end(); ++it) {
                const std::map<int, float> row = train_data.get_row(*it);
                auto element = row.find(l);
                _errors.at(*it) += delta * element->second;
            }
        }
        for (int l = 0; l < _max_feature + 1; ++l) {
            for (int f = 0; f < _k_2; ++f) {
                double delta = -_v.at(l).at(f);
                double _v_star = 0.0;
                double h_square = 0.0;
                const std::vector <int> &valid_objects = train_data.get_feature_objects(l);
                for (auto it = valid_objects.begin(); it != valid_objects.end(); ++it) {
                    const std::map<int, float> row = train_data.get_row(*it);
                    auto element = row.find(l);
                    double h = element->second*(_cache.at(*it).at(f) - _v.at(l).at(f)*element->second);
                    _v_star += h * (_v.at(l).at(f) * h - _errors.at(*it));
                    h_square += h*h;
                }
                _v_star /= h_square + _reg_v;
                delta += _v_star;
                _v.at(l).at(f) = _v_star;
                for (auto it = valid_objects.begin(); it != valid_objects.end(); ++it) {
                    const std::map<int, float> row = train_data.get_row(*it);
                    auto element = row.find(l);
                    _errors.at(*it) += delta * element->second;
                    _cache.at(*it).at(f) += delta * element->second;
                }
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
        if (_learning_method == ALS) {
            _errors = std::vector<double> (train_data.size(), 0.0);
            _cache = std::vector<std::vector<double> >(train_data.size(), std::vector<double>(_k_2, 0.0));
        }
        for (int i = 0; i < _iterations; i++) {
            learn_step(train_data, test_data, i);
        }
    }

private:
    float _learning_rate;
    int _iterations;
    learning_method _learning_method;
    task_type _task_type;
    bool _k_0 = true;
    bool _k_1 = true;
    int _k_2 = 4;
    double _w_0 = 0.0;
    int _max_feature;
    float _min_target;
    float _max_target;
    double _reg_w0 = 0.0;
    double _reg_w = 0.0;
    double _reg_v = 0.0;
    std::vector<double> _w;
    std::vector<std::vector<double> > _v;
    std::vector<double> _linear_vx_sum;
    std::vector<double> _errors;
    std::vector<std::vector<double> > _cache;
    bool first_traverse = true;
};
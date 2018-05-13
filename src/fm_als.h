//
// Created by Адель Хафизова on 13.05.18.
//

#ifndef FM_FM_ALS_H
#define FM_FM_ALS_H

#endif //FM_FM_ALS_H
//
// Created by Адель Хафизова on 26.04.18.
//
#include <random>


class FactorizationMachineALS: public FactorizationMachine {
public:
    FactorizationMachineALS(float lr, const std::string &reg_const, int num_iter, const TaskType &type, int max_feature,
                            float max_target, float min_target, int train_size) {
        FactorizationMachine(lr, reg_const, num_iter, type, max_feature, max_target, min_target);
        _errors = std::vector<double> (train_size, 0.0);
        _cache = std::vector<std::vector<double> >(train_size, std::vector<double>(_k_2, 0.0));
    };

    void learn_step(Dataset *external_train_data) override {
        MemoryDataset* train_data = dynamic_cast<MemoryDataset*>(external_train_data);
        if (first_traverse) {
            for (int i = 0; i < train_data->size(); ++i) {
                float y_hat = predict(train_data);
                _errors.at(i) = y_hat - (double)train_data->get_target();
                auto row = train_data->get_row();
                for (int f = 0; f < _k_2; f++) {
                    _cache.at(i).at(f) = 0.0;
                    for (auto it = row.begin(); it != row.end(); it++) {
                        _cache.at(i).at(f) += _v.at(it->first).at(f)*it->second;
                    }
                }
            }
            first_traverse = false;
            train_data->next_row();
        }
        als_step(train_data);
    }

    double predict(const Dataset *external_train_data) override {
        MemoryDataset* train_data = dynamic_cast<MemoryDataset*>(external_train_data);
        float y_hat = 0.0;
        const std::map<int, float> row = train_data->get_row();

        for (int f = 0; f < _k_2; f++) {
            double linear_sum = 0.0;
            double sum_of_squared = 0.0;
            for (auto it = row.begin(); it != row.end(); it++) {
                double sum_element = _v.at(it->first).at(f)*it->second;
                linear_sum += sum_element;
                sum_of_squared += sum_element*sum_element;
            }
            y_hat += linear_sum*linear_sum - sum_of_squared;
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
        return y_hat;
    }

    void als_step(const MemoryDataset *train_data) {
        double _errors_sum = 0.0;
        for (int i = 0; i < train_data->size(); ++i) {
            _errors_sum += _errors.at(i);
        }
        double delta = -_w_0;
        _w_0 = (_w_0 * train_data->size() - _errors_sum) / (train_data->size() + _reg_w0);
        delta += _w_0;
        for (int i = 0; i < train_data->size(); ++i) {
            _errors.at(i) += delta;
        }
        for (int l = 0; l < _max_feature + 1; ++l) {
            const std::vector <int> &valid_objects = train_data->get_feature_objects(l);
            if (valid_objects.size() == 0) {
                continue;
            }
            double delta = -_w.at(l);
            double _w_l_star = 0.0;
            double x_l_square = 0.0;
            for (auto it = valid_objects.begin(); it != valid_objects.end(); ++it) {
                const std::map<int, float> row = train_data->get_row();
                auto element = row.find(l);
                _w_l_star += element->second * (_w.at(l) * element->second - _errors.at(*it));
                x_l_square += element->second * element->second;
            }
            if (x_l_square > 0.0) {
                _w_l_star /= x_l_square + _reg_w;
            }
            delta += _w_l_star;
            _w.at(l) = _w_l_star;
            for (auto it = valid_objects.begin(); it != valid_objects.end(); ++it) {
                const std::map<int, float> row = train_data->get_row(*it);
                auto element = row.find(l);
                _errors.at(*it) += delta * element->second;
            }
        }
        for (int f = 0; f < _k_2; ++f) {
            for (int l = 0; l < _max_feature + 1; ++l) {
                const std::vector <int> &valid_objects = train_data->get_feature_objects(l);
                if (valid_objects.size() == 0) {
                    continue;
                }
                double delta = -_v.at(l).at(f);
                double v_old = _v.at(l).at(f);
                double _v_star = 0.0;
                double h_square = 0.0;
                for (auto it = valid_objects.begin(); it != valid_objects.end(); ++it) {
                    const std::map<int, float> row = train_data->get_row(*it);
                    auto element = row.find(l);
                    double h = element->second * (_cache.at(*it).at(f) - _v.at(l).at(f) * element->second);
                    _v_star += h * (_v.at(l).at(f) * h - _errors.at(*it));
                    h_square += h * h;
                }
                _v_star /= (h_square + _reg_v);
                delta += _v_star;
                _v.at(l).at(f) = _v_star;
                for (auto it = valid_objects.begin(); it != valid_objects.end(); ++it) {
                    const std::map<int, float> row = train_data->get_row(*it);
                    auto element = row.find(l);
                    double h = element->second * (_cache.at(*it).at(f) - element->second * v_old);
                    _errors.at(*it) += delta * h;
                    _cache.at(*it).at(f) += delta * element->second;
                }
            }
        }
    }

    /*double evaluate(const Dataset *external_data) override {
        MemoryDataset* data = dynamic_cast<MemoryDataset*>(external_data);
        double error = 0.0;
        for (int i = 0; i < data->size() - 1; ++i) {
            float y_hat = predict(data);
            y_hat = std::max(y_hat, _min_target);
            y_hat = std::min(y_hat, _max_target);
            double target = (double) data->get_target();
            if (_task_type == classification) {
                //TODO: check that this function is the correct way to measure
                error += -target*log(y_hat) - (1 - target)*log(1 - y_hat);
            } else {
                error += pow(y_hat - target, 2);
            }
        }
        if (_task_type == classification) {
            return error/data->size();
        } else {
            return sqrt(error/data->size());
        }
    }*/

private:
    float _learning_rate;
    int _iterations;
    TaskType _task_type;
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
    std::vector<double> _errors;
    std::vector<std::vector<double> > _cache;
    bool first_traverse = true;
};
//
// Created by Адель Хафизова on 26.04.18.
//

#ifndef FM_FACTORIZATION_MACHINE_H
#define FM_FACTORIZATION_MACHINE_H

#endif //FM_FACTORIZATION_MACHINE_H
#include <random>
#include <iostream>


class FactorizationMachine {
public:
    FactorizationMachine() = default;
    explicit FactorizationMachine(float lr, const std::string &reg_const, int num_iter,
                                  const TaskType &type, int max_feature, float max_target, float min_target);;

    virtual void learn_step(Dataset *train_data) = 0;
    virtual double predict(const Dataset *data) = 0;
    double evaluate(const Dataset *data);
    void launch_learning(Dataset *train_dat, Dataset *test_data);

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
    std::vector<double> _linear_vx_sum;
    std::vector<double> _errors;
    std::vector<std::vector<double> > _cache;
    bool first_traverse = true;
};
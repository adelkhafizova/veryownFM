//
// Created by Адель Хафизова on 13.05.18.
//

#ifndef FM_FM_SGD_H
#define FM_FM_SGD_H

#endif //FM_FM_SGD_H

//
// Created by Адель Хафизова on 26.04.18.
//

#include <random>
#include "factorization_machine.h"


class FactorizationMachineSGD: public FactorizationMachine {
public:
    FactorizationMachineSGD(float lr, const std::string &reg_const, int num_iter, const TaskType &type, int max_feature,
                            float max_target, float min_target);;

    void learn_step(Dataset *train_data) override;
    double predict(Dataset const *data) override;
    void sgd_step(const std::map<int, float> &row, double coefficient);

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
};
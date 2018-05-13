//
// Created by Адель Хафизова on 13.05.18.
//

#ifndef FM_FM_SGD_H
#define FM_FM_SGD_H

#include "factorization_machine.h"


class FactorizationMachineSGD: public FactorizationMachine {
public:
    FactorizationMachineSGD(float lr, const std::string &reg_const, int num_iter, const TaskType &type, bool use_bias,
                            bool use_linear, int pairwise_rank, int max_feature, float max_target, float min_target);

    void learn_step(Dataset *train_data) override;
    double predict(Dataset const *data) override;
    void sgd_step(const std::map<int, float> &row, double coefficient);

private:
    std::vector<double> _linear_vx_sum;
};

#endif //FM_FM_SGD_H

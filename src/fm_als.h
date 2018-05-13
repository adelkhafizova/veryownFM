//
// Created by Адель Хафизова on 13.05.18.
//

#ifndef FM_FM_ALS_H
#define FM_FM_ALS_H
#include "factorization_machine.h"


class FactorizationMachineALS: public FactorizationMachine {
public:
    FactorizationMachineALS(float lr, const std::string &reg_const, int num_iter, const TaskType &type, bool use_bias,
                            bool use_linear, int pairwise_rank, int max_feature, float max_target, float min_target,
                            int train_size);

    void learn_step(Dataset *external_train_data) override;
    double predict(const Dataset *external_train_data) override;
    double predict(const Dataset *external_train_data, int i);
    void als_step(const MemoryDataset *train_data);

protected:
    std::vector<double> _errors;
    std::vector<std::vector<double> > _cache;
    bool first_traverse = true;
};

#endif //FM_FM_ALS_H
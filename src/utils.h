//
// Created by Адель Хафизова on 26.04.18.
//

#ifndef FM_UTILS_H
#define FM_UTILS_H

#include <fstream>
#include <climits>
#include <random>
#include <map>


enum LearningMethod {
    SGD = 1,
    ALS
};


enum StorageType {
    memory = 1,
    inplace
};


enum TaskType {
    classification = 1,
    regression
};


struct HashFunctionParams {
    static const int kDefaultModuloPrime = 2038074743;
    int64_t hash_multiplier;
    int64_t hash_shift;
    int value_size;
    HashFunctionParams() {};
    explicit HashFunctionParams(std::mt19937 &generator, int value_size);
    int HashFunction(int key) const;
};


class Dataset {
public:
    Dataset() = default;
    explicit Dataset(const std::string &filename, HashFunctionParams &hash_params);
    virtual void next_row() = 0;
    virtual const std::map<int, float> &get_row() const = 0;
    virtual const float &get_target() const = 0;
    int get_max_feature() const {
        return _max_feature;
    }
    float get_max_target() const;
    float get_min_target() const;
    int size() const;;
    virtual ~Dataset() = default;

protected:
    std::string _filename;
    HashFunctionParams _hash_params;
    bool use_hashing;
    int row_number;
    int _max_feature;
    float _max_target;
    float _min_target;
    virtual void set_row(const std::map<int, float> &current_row) = 0;
    virtual void set_target(float current_target) = 0;
    void parse_line(const std::string &line, std::map<int, float> *features_row, float &target);
    void preprocessing();
};


class MemoryDataset : public Dataset {
public:
    MemoryDataset(const std::string &filename, HashFunctionParams &hash_params);
    void next_row() override;
    const std::map<int, float> &get_row() const override;
    const std::map<int, float> &get_row(int i) const;
    const float &get_target() const override;
    const float &get_target(int i) const;
    const std::vector<int> &get_feature_objects(int l) const;

private:
    std::vector<std::map<int, float>> _data;
    std::vector<float> _target;
    std::vector<std::vector<int>> _feature_objects;
    void set_row(const std::map<int, float> &current_row) override;
    void set_target(float current_target) override;
    unsigned long current_row_index = 0;
};


class IterDataset : public Dataset {
public:
    IterDataset(const std::string &filename, HashFunctionParams &hash_params);;
    void next_row() override;
    const std::map<int, float> &get_row() const override;
    const float &get_target() const override;

private:
    std::ifstream file;
    std::map<int, float> _memory_row;
    float _memory_target;
    void set_row(const std::map<int, float> &current_row) override;
    void set_target(float current_target) override;
};

#endif //FM_UTILS_H

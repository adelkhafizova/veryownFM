//
// Created by Адель Хафизова on 13.05.18.
//

#include <iostream>
#include "utils.h"

MemoryDataset::MemoryDataset(const std::string &filename, HashFunctionParams &hash_params) : Dataset(filename,
                                                                                                     hash_params) {
    preprocessing();
    _feature_objects = std::vector<std::vector<int>>(_max_feature + 1, std::vector<int>());
    for (int i = 0; i < _data.size(); ++i) {
        auto row = _data.at(i);
        for (auto it = row.begin(); it != row.end(); ++it) {
            _feature_objects.at(it->first).push_back(i);
        }
    }
};

void MemoryDataset::next_row() {
    current_row_index = (current_row_index + 1) % size();
}

const std::map<int, float> &MemoryDataset::get_row() const {
    return _data.at(current_row_index);
}

const std::map<int, float> &MemoryDataset::get_row(int i) const {
    return _data.at(i);
}

const float &MemoryDataset::get_target() const {
    return _target.at(current_row_index);
}

const float &MemoryDataset::get_target(int i) const {
    return _target.at(i);
}

const std::vector<int> &MemoryDataset::get_feature_objects(int l) const {
    return _feature_objects.at(l);
}

void MemoryDataset::set_row(const std::map<int, float> &current_row) {
    if (use_hashing) {
        std::map<int, float> hashed_row;
        for (const auto &elem: current_row) {
            int new_key = _hash_params.HashFunction(elem.first);
            hashed_row[new_key] += elem.second;
        }
        _data.push_back(hashed_row);
    } else {
        _data.push_back(current_row);
    }
}

void MemoryDataset::set_target(float current_target) {
    _target.push_back(current_target);
}


void IterDataset::next_row() {
    std::string line;
    if (!getline(file, line)) {
        file.close();
        file.open(_filename);
        getline(file, line);
    }
    std::map<int, float> current_row = std::map<int, float>();
    float current_target;
    parse_line(line, &current_row, current_target);
    set_row(current_row);
    set_target(current_target);
};

void IterDataset::set_row(const std::map<int, float> &current_row) {
    if (use_hashing) {
        std::map<int, float> hashed_row;
        for (const auto &elem: current_row) {
            int new_key = _hash_params.HashFunction(elem.first);
            hashed_row[new_key] += elem.second;
        }
        _memory_row = hashed_row;
    } else {
        _memory_row = current_row;
    }
}

void IterDataset::set_target(float current_target) {
    _memory_target = current_target;
}

IterDataset::IterDataset(const std::string &filename, HashFunctionParams &hash_params) : Dataset(filename,
                                                                                                 hash_params) {
    preprocessing();
    file.open(_filename);
    next_row();
}

const std::map<int, float> &IterDataset::get_row() const {
    return _memory_row;
}

const float &IterDataset::get_target() const {
    return _memory_target;
}


Dataset::Dataset(const std::string &filename, HashFunctionParams &hash_params) {
    _filename = filename;
    _hash_params = hash_params;
    use_hashing = hash_params.value_size > 0;
    row_number = 0;
    _max_feature = 0;
    _max_target = INT64_MIN;
    _min_target = INT64_MAX;
};


void Dataset::preprocessing() {
    std::ifstream file;
    file.open(_filename);
    std::string line;
    std::cout << "Preprocessing" << std::endl;
    while (getline(file, line)) {
        row_number += 1;
        float target;
        std::map<int, float> row;
        parse_line(line, &row, target);
        set_row(row);
        set_target(target);

        _max_target = std::max(target, _max_target);
        _min_target = std::min(target, _min_target);

        if (use_hashing)
            _max_feature = _hash_params.value_size - 1;
        else {
            _max_feature = std::max(_max_feature, row.rbegin()->first);
        }
    }
    std::cout << "Processed " << size() << " rows" << std::endl;
    std::cout << "Target from " << _min_target << " to " << _max_target << std::endl;
    std::cout << "Max feature index is " << _max_feature << std::endl;
    file.close();
}

void Dataset::parse_line(const std::string &line, std::map<int, float> *features_row, float &target) {
    const char *pline = line.c_str();
    int shift;
    sscanf(pline, "%f%n", &target, &shift);
    pline += shift;

    int feature_id;
    float value;
    while (sscanf(pline, "%d:%f%n", &feature_id, &value, &shift) > 1) {
        pline += shift;
        (*features_row)[feature_id] = value;
    }
}

float Dataset::get_max_target() const {
    return _max_target;
}

float Dataset::get_min_target() const {
    return _min_target;
}

int Dataset::size() const {
    return row_number;
}

HashFunctionParams::HashFunctionParams(std::mt19937 &generator, int _value_size) {
    value_size = _value_size;
    std::uniform_int_distribution<> a_dis(1, kDefaultModuloPrime - 1);
    hash_multiplier = a_dis(generator);
    std::uniform_int_distribution<> b_dis(0, kDefaultModuloPrime - 1);
    hash_shift = b_dis(generator);
}

int HashFunctionParams::HashFunction(int key) const {
    if (value_size == 1) {
        return 0;
    }
    int64_t sum = (hash_multiplier * key + hash_shift) % kDefaultModuloPrime;
    if (sum < 0) {
        sum += kDefaultModuloPrime;
    }
    return static_cast<int>(sum) % value_size;
}


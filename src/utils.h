//
// Created by Адель Хафизова on 26.04.18.
//

#ifndef FM_UTILS_H
#define FM_UTILS_H

#endif //FM_UTILS_H

#include <fstream>
#include <climits>


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

    HashFunctionParams(){};
    explicit HashFunctionParams(std::mt19937 &generator, const int value_size);

    int HashFunction(const int key) const;
};


class Dataset {
public:
    Dataset () = default;
    explicit Dataset(const std::string &filename, HashFunctionParams &hash_params);

    virtual void next_row() = 0;

    virtual const std::map<int, float>& get_row() const = 0;
    virtual const float & get_target() const = 0;
    virtual const std::vector<int> & get_feature_objects(int l) const = 0;

    int get_max_feature() const {
        return _max_feature;
    }
    float get_max_target() const {
        return _max_target;
    }
    float get_min_target() const {
        return _min_target;
    }

    int size() const {
        return row_number;
    };

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

    void parse_line(const std::string &line, std::map<int, float> *features_row, float& target);
    void preprocessing();
};


class MemoryDataset:  public Dataset {
public:
    MemoryDataset(const std::string &filename, HashFunctionParams &hash_params) : Dataset(filename, hash_params)  {

        preprocessing();
        _feature_objects = std::vector<std::vector<int> >(_max_feature + 1, std::vector<int>());
        for (int i = 0; i < _data.size(); ++i) {
            auto row = _data.at(i);
            for (auto it = row.begin(); it != row.end(); ++it) {
                _feature_objects.at(it->first).push_back(i);
            }
        }
    };

    void next_row() {
        current_row_index = (current_row_index + 1) % size();
    }

    const std::map<int, float>& get_row() const{
        return _data.at(current_row_index);
    }

    const float & get_target() const {
        return _target.at(current_row_index);
    }

    const std::vector<int> & get_feature_objects(int l) const {
        return _feature_objects.at(l);
    }

private:
    unsigned long current_row_index = 0;
    std::vector<std::map<int, float> > _data;
    std::vector<float> _target;
    std::vector<std::vector<int> > _feature_objects;

    void set_row(const std::map<int, float>& current_row);

    void set_target(float current_target) {
        _target.push_back(current_target);
    }
};


class IterDataset : public Dataset{

public:
    IterDataset(const std::string &filename, HashFunctionParams &hash_params) : Dataset(filename, hash_params)  {
        preprocessing();
        file.open(_filename);
        next_row();
    };

    void next_row();

    const std::map<int, float>& get_row() const{
        return _memory_row;
    }

    const float & get_target() const {
        return _memory_target;
    };

    // fake method
    const std::vector<int> & get_feature_objects(int l) const {
        return std::vector<int>();
    }

private:
    std::ifstream file;

    std::map<int, float> _memory_row;
    float _memory_target;

    void set_row(const std::map<int, float>& current_row) {
        if (use_hashing) {
            std::map<int, float> hashed_row;
            for (auto elem: current_row) {
                int new_key = _hash_params.HashFunction(elem.first);
                hashed_row[new_key] += elem.second;
            }
            _memory_row = hashed_row;
        } else {
            _memory_row = current_row;
        }
    }

    void set_target(float current_target) {
        _memory_target = current_target;
    }
};


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
    std::cout << "Preprocessing\n";
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

void Dataset::parse_line(const std::string &line, std::map<int, float> *features_row, float& target) {
    const char *pline = line.c_str();
    int shift;
    sscanf(pline, "%f%n", &target, &shift);
    pline += shift;

    int feature_id;
    float value;
    while(sscanf(pline, "%d:%f%n", &feature_id, &value, &shift) > 1) {
        pline += shift;
        (*features_row)[feature_id] = value;
    }
}

void MemoryDataset::set_row(const std::map<int, float>& current_row) {
    if (use_hashing) {
        std::map<int, float> hashed_row;
        for (auto elem: current_row) {
            int new_key = _hash_params.HashFunction(elem.first);
            hashed_row[new_key] += elem.second;
        }
        _data.push_back(hashed_row);
    } else {
        _data.push_back(current_row);
    }
}

void IterDataset::next_row() {
    std::string line;
    if (!getline(file, line)) {
        file.close();
        file.open(_filename);
        getline(file, line);
    }
    std::map<int, float>current_row = std::map<int, float>();
    float current_target;
    parse_line(line, &current_row, current_target);
    set_row(current_row);
    set_target(current_target);
};

HashFunctionParams::HashFunctionParams(std::mt19937 &generator, const int _value_size) {
    value_size = _value_size;
    std::uniform_int_distribution<> a_dis(1, kDefaultModuloPrime - 1);
    hash_multiplier = a_dis(generator);
    std::uniform_int_distribution<> b_dis(0, kDefaultModuloPrime - 1);
    hash_shift = b_dis(generator);
}

int HashFunctionParams::HashFunction(const int key) const {
    if (value_size == 1) {
        return 0;
    }
    int64_t sum = (hash_multiplier * key + hash_shift) % kDefaultModuloPrime;
    if (sum < 0) {
        sum += kDefaultModuloPrime;
    }
    return sum % value_size;
}

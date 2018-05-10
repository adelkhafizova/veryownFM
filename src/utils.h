//
// Created by Адель Хафизова on 26.04.18.
//

#ifndef FM_UTILS_H
#define FM_UTILS_H

#endif //FM_UTILS_H

#include <fstream>
#include <climits>


enum learning_method {
    SGD = 1,
    ALS
};

enum task_type {
    classification = 1,
    regression
};

//class Dataset {
//public:
//    Dataset(const std::string &filename) {
//        std::ifstream file;
//        file.open(filename);
//        std::string s;
//
//        while (getline(file, s)) {
//            float target;
//            std::map<int, float> m;
//            const char *pline = s.c_str();
//            int shift;
//            sscanf(pline, "%f%n", &target, &shift);
//            pline += shift;
//            _target.push_back(target);
//            _max_target = std::max(target, _max_target);
//            _min_target = std::min(target, _min_target);
//            int feature_id;
//            float value;
//            while (sscanf(pline, "%d:%f%n", &feature_id, &value, &shift) > 1) {
//                pline += shift;
//                m[feature_id] = value;
//                _max_feature = std::max(feature_id, _max_feature);
//
//            }
//            _data.push_back(m);
//#ifdef DEBUG
//            if (_data.size() % 10000 == 0) {
//                std::cout << _data.size() << std::endl;
//            }
//#endif
//        }
//        std::cout << _data.size() << " " << _target.size() << std::endl;
//        _feature_objects = std::vector<std::vector<int> >(_max_feature + 1, std::vector<int>());
//        for (int i = 0; i < _data.size(); ++i) {
//            auto row = _data.at(i);
//            for (auto it = row.begin(); it != row.end(); ++it) {
//                _feature_objects.at(it->first).push_back(i);
//            }
//        }
//
//    };
////    const std::map<int, float> & get_row(int i) const {
//    const std::map<int, float>& get_row() const{
//        return _data.at(current_row_index);
//    };
//
//    void next_row() {
//        current_row_index = (current_row_index + 1) % size();
//    };
//
//    const int get_max_feature() const {
//        return _max_feature;
//    }
//
//    const float get_max_target() const {
//        return _max_target;
//    }
//
//    const int get_min_target() const {
//        return _min_target;
//    }
//
//    const int size() const {
//        return _data.size();
//    }
//
//    const float & get_target() const {
//        return _target.at(current_row_index);
//    };
//
//    const std::vector<int> & get_feature_objects(int l) const {
//        return _feature_objects.at(l);
//    }
//
//private:
//    unsigned long current_row_index = 0;
//
//    std::vector<std::map<int, float> > _data;
//    std::vector<float> _target;
//    int _max_feature = 0;
//    float _max_target = INT64_MIN;
//    float _min_target = INT64_MAX;
//    std::vector<std::vector<int> > _feature_objects;
//};
class Dataset {
public:
    Dataset(const std::string &input_filename) {
        filename = input_filename;

        preprocessing();

        file.open(input_filename);
        next_row();
    };


    const std::map<int, float> get_row() const {
        return current_row;
    }

    void next_row() {
        std::string line;
        if (!getline(file, line)) {
            file.close();
            file.open(filename);
            getline(file, line);
        }
        current_row = std::map<int, float>();
        parse_line(line, current_row, current_target);
    };


    const int get_max_feature() const {
        return _max_feature;
    }

    const float get_max_target() const {
        return _max_target;
    }

    const float get_min_target() const {
        return _min_target;
    }

    const int size() const {
        return row_number;
    }

    const float & get_target() const {
        return current_target;
    };

    const std::vector<int> & get_feature_objects(int l) const {
        return std::vector<int>();
    }
private:
    std::map<int, float> current_row;
    float current_target;

    std::string filename;
    std::ifstream file;

    int _max_feature = 0;
    int row_number = 0;

    float _max_target = INT64_MIN;
    float _min_target = INT64_MAX;

    void preprocessing() {
        file.open(filename);
        std::cout << "Preprocessing\n";
        std::string line;
        while(getline(file, line)) {
            row_number += 1;
            current_row = std::map<int, float>();
            parse_line(line, current_row, current_target);
            _max_target = std::max(current_target, _max_target);
            _min_target = std::min(current_target, _min_target);
            _max_feature = std::max(_max_feature, current_row.rbegin()->first);
        }
        std::cout << "Processed " << row_number << " rows" << std::endl;
        std::cout << "Target from " << _min_target << " to " << _max_target << std::endl;
        std::cout << "Max feature index is " << _max_feature << std::endl;
        file.close();
    }

    void parse_line(const std::string &line, std::map<int, float> &features_row, float& target) {
        const char *pline = line.c_str();
        int shift;
        sscanf(pline, "%f%n", &target, &shift);
        pline += shift;

        int feature_id;
        float value;
        while (sscanf(pline, "%d:%f%n", &feature_id, &value, &shift) > 1) {
            pline += shift;
            features_row[feature_id] = value;
            _max_feature = std::max(feature_id, _max_feature);
        }
    }
};
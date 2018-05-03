#include <iostream>
#include "cxxopts/include/cxxopts.hpp"
#include "src/utils.h"
#include "src/factorization_machine.h"


int main(int argc, const char *argv[]) {
    float learning_rate = 0.1;
    float regularization_const = 0;
    int iterations = 30;
    learning_method method = SGD;
    std::string train_filename;
    std::string test_filename;
    std::string out_filename;
    task_type type;
    try {
        cxxopts::Options options("Factorization machines", "Library for using factorization algorithm");
        options
                .positional_help("[optional args]")
                .show_positional_help();
        options.add_options()
                ("l,learning_rate", "Learning rate value, default 0.1", cxxopts::value<float>())
                ("r,regularization_const", "Regularization constant, default 0", cxxopts::value<float>())
                ("i,iterations", "Number of iterations, default 100", cxxopts::value<int>())
                ("m,learning_method", "Learning method (SGD, ALS), default SGD", cxxopts::value<std::string>())
                ("t,train_filename", "Training file name", cxxopts::value<std::string>())
                ("e,test_filename", "Testing file name", cxxopts::value<std::string>())
                ("o,out_filename", "Output file name", cxxopts::value<std::string>())
                ("s,task_type", "Task type parameter", cxxopts::value<std::string>())
                ("h,help", "Usage description");
        auto result = options.parse(argc, argv);

        if (result.count("help"))
        {
            std::cout << options.help() << std::endl;
            exit(0);
        }

        if (result.count("l")) {
            learning_rate = result["l"].as<float>();
        }
        if (result.count("r")) {
            regularization_const = result["r"].as<float>();
        }
        if (result.count("i")) {
            iterations = result["i"].as<int>();
        }
        if (result.count("m")) {
            if (result["m"].as<std::string>() == "SGD") {
                method = SGD;
            } else {
                if (result["m"].as<std::string>() == "ALS") {
                method = ALS;
                } else {
                    throw cxxopts::OptionException("Bad learning method definition");
                }
            }
        }
        if (result.count("i")) {
            iterations = result["i"].as<int>();
        }
        if (result.count("t")) {
            train_filename = result["t"].as<std::string>();
        } else {
            throw cxxopts::OptionException("No train file");
        }
        if (result.count("e")) {
            test_filename = result["e"].as<std::string>();
        } else {
            throw cxxopts::OptionException("No test file");
        }
        if (result.count("o")) {
            out_filename = result["o"].as<std::string>();
        } else {
            throw cxxopts::OptionException("No output file");
        }
        if (result.count("s")) {
            if (result["s"].as<std::string>() == "regression") {
                type = regression;
            } else {
                if (result["s"].as<std::string>() == "classification") {
                    type = classification;
                } else {
                    throw cxxopts::OptionException("Bad task type definition");
                }
            }
        }
    } catch (const cxxopts::OptionException &e) {
        std::cout << "error parsing options: " << e.what() << std::endl;
        exit(1);
    }
    Dataset train_dataset(train_filename);
    Dataset test_dataset(test_filename);
    int max_feature = std::max(train_dataset.get_max_feature(), test_dataset.get_max_feature());
    FactorizationMachine factorizationMachine(learning_rate, regularization_const,
                                              iterations, method, type, max_feature,
                                              std::max(train_dataset.get_max_target(), test_dataset.get_max_target()),
                                              std::min(train_dataset.get_min_target(), test_dataset.get_min_target()));
    factorizationMachine.launch_learning(train_dataset, test_dataset);
    return 0;
}
#include <iostream>
#include <random>
#include "cxxopts/include/cxxopts.hpp"
#include "src/utils.h"
#include "src/factorization_machine.h"
#include "src/fm_sgd.h"
#include "src/fm_als.h"


int main(int argc, const char *argv[]) {
    float learning_rate = 0.1;

    std::string regularization_const;
    int iterations = 10;
    LearningMethod method = SGD;
    StorageType storage_type = memory;
    std::string train_filename;
    std::string test_filename;
    std::string out_filename;
    bool use_bias = true;
    bool use_linear = true;
    int pairwise_rank = 4;
    TaskType type;

    int hash_size = -1;
    int random_seed = 42;

    try {
        cxxopts::Options options("Factorization machines", "Library for using factorization algorithm");
        options
                .positional_help("[optional args]")
                .show_positional_help();
        options.add_options()
                ("l,learning_rate", "Learning rate value, default 0.1", cxxopts::value<float>())
                ("bias", "Use bias or not", cxxopts::value<bool>())
                ("linear", "Use linear or not", cxxopts::value<bool>())
                ("pairwise", "Two-way interactions order", cxxopts::value<int>())
                ("r,regularization_const", "Regularization constant, default 0", cxxopts::value<std::string>())
                ("i,iterations", "Number of iterations, default 100", cxxopts::value<int>())
                ("m,learning_method", "Learning method (SGD, ALS), default SGD", cxxopts::value<std::string>())
                ("g,inplace", "Storage (inplace, memory), default memory", cxxopts::value<std::string>())
                ("t,train_filename", "Training file name", cxxopts::value<std::string>())
                ("e,test_filename", "Testing file name", cxxopts::value<std::string>())
                ("s,task_type", "Task type parameter", cxxopts::value<std::string>())
                ("hash_size", "Positive hash size if use hashing trick, else -1, default -1", cxxopts::value<int>())
                ("hash_random_seed", "Random seed of hashing", cxxopts::value<int>())
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
        if (result.count("bias")) {
            use_bias = result["bias"].as<bool>();
        }
        if (result.count("linear")) {
            use_linear = result["linear"].as<bool>();
        }
        if (result.count("pairwise")) {
            pairwise_rank = result["pairwise"].as<int>();
        }
        if (result.count("r")) {
            regularization_const = result["r"].as<std::string>();
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
        if (result.count("g")) {
            if (result["g"].as<std::string>() == "inplace") {
                storage_type = inplace;
                if (method == ALS) {
                    throw cxxopts::OptionException("Unsupported data storage for ALS, memory only");
                }
            } else {
                if (result["g"].as<std::string>() == "memory") {
                    storage_type = memory;
                } else {
                    throw cxxopts::OptionException("Bad storage definition");
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
        if (result.count("s")) {
            if (result["s"].as<std::string>() == "regression") {
                type = regression;
            } else {
                if (result["s"].as<std::string>() == "classification") {
                    type = classification;
                    if (method == ALS) {
                        throw cxxopts::OptionException("Unsupported task type for ALS, regression only");
                    }
                } else {
                    throw cxxopts::OptionException("Bad task type definition");
                }
            }
        }
        if (result.count("hash_size")) {
            hash_size = result["hash_size"].as<int>();
        }
        if (result.count("hash_random_seed")) {
            random_seed = result["hash_random_seed"].as<int>();
        }

    } catch (const cxxopts::OptionException &e) {
        std::cout << "error parsing options: " << e.what() << std::endl;
        exit(1);
    }

    Dataset *train_dataset;
    Dataset *test_dataset;

    std::mt19937 generator(random_seed);

    HashFunctionParams hash_params = HashFunctionParams(generator, hash_size);

    if (storage_type == memory){
        train_dataset = new MemoryDataset(train_filename, hash_params);
        test_dataset  = new MemoryDataset(test_filename, hash_params);
    } else {
        train_dataset = new IterDataset(train_filename, hash_params);
        test_dataset  = new IterDataset(test_filename, hash_params);
    }
    FactorizationMachine *factorizationMachine;
    if (method == SGD) {
        factorizationMachine = new FactorizationMachineSGD(learning_rate, regularization_const,
                                                           iterations, type, use_bias, use_linear, pairwise_rank,
                                                           std::max(train_dataset->get_max_feature(),
                                                                    test_dataset->get_max_feature()),
                                                           train_dataset->get_max_target(),
                                                           train_dataset->get_min_target());
    } else {
        factorizationMachine = new FactorizationMachineALS(learning_rate, regularization_const,
                                                           iterations, type, use_bias, use_linear, pairwise_rank,
                                                           std::max(train_dataset->get_max_feature(),
                                                                    test_dataset->get_max_feature()),
                                                           train_dataset->get_max_target(),
                                                           train_dataset->get_min_target(), train_dataset->size());
    }
    factorizationMachine->launch_learning(train_dataset, test_dataset);
    delete train_dataset;
    delete test_dataset;
    delete factorizationMachine;
    return 0;
}
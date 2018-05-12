# veryownFM
Factorization machines implementation
==========================================
This is a C++ implementation of factorization machines algorithm

It supports:

* dense and sparse inputs
* two optimization methods: stochastic gradient descent and alternating least-squares
* classification/regression via logistic and mse loss functions
* gradient clipping in SGD

# Usage
This is a command-line tool, CMakeLists is provided for building an executable from source code
Command line arguments:
* learning_rate: necessary for SGD, default 0.1
* regularization_const: one or three values for each weights type (bias, linear and pairwise)
* iterations: number of iterations to be launched
* learning_method: either SGD or ALS
* inplace: whether to store a training dataset in memory or not (inplace, memory), default memory
* train_filename: path to train dataset in lightsvm format
* test_filename: path to test dataset in lightsvm format
* task_type: regression or classification
* hash_size: positive hash size if use hashing trick, else -1, default -1
* hash_random_seed: random seed of hashing
* help: command-line arguments description

# Example
./FM --task_type regression --train_filename train.libfm --test_filename test.libfm --learning_method ALS
./FM --task_type classification --train_filename train.libfm --test_filename test.libfm --learning_rate 0.000001 --learning_method SGD

# Tests
The utility has been tested on the following datasets:
* MovieLens
* Avazu

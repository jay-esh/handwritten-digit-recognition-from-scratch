#include <stdio.h>
#include <fstream> // for file access
#include <iostream>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <vector>
#include <tuple>
#include <cmath>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using namespace std;

// pseudo random number generator
struct xorshift
{
    unsigned x, y, z, w;

    xorshift() : x(123456789), y(38012123), z(7777777), w(8392032) {}

    unsigned next()
    {
        unsigned t = x ^ (x << 11);
        x = y;
        y = z;
        z = w;
        return w = w ^ (w >> 19) ^ t ^ (t >> 8);
    }
} rng;

double rand01()
{
    if (rng.next() % 2)
    {
        return (-((rng.next() % 1000000001) / 1000000000.0));
    }

    return ((rng.next() % 1000000001) / 1000000000.0);
}
// random number generation for weights
// weights is a vector of matrices
// each matrix of this vector will have the size of
// (764 x 16) -> (16 x 16) -> (16 x 10)
MatrixXd randomNumberGenW(int row, int column)
{
    MatrixXd weights(row, column);
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < column; j++)
        {
            weights(i, j) = rand01();
        }
    }
    return weights;
}

//////////////////////////////////////////////////////////////

// random number generation for biases
// the biases are also a vector of matrices
// the sizes are:
// (1 x 16) -> (1 x 16) -> (1 x 10)
MatrixXd randomNumberGenB(int column)
{
    MatrixXd biases(1, column);
    for (int i = 0; i < column; i++)
    {
        srand(time(NULL));
        biases(0, i) = rand01();
    }
    return biases;
}

//////////////////////////////////////////////////////////////

// sigmoid function to get a number between 0 and 1
// inputs a matrix and outputs a matrix
// sigmoid function is run on all the elements of the input matrix
MatrixXd sigmoid(MatrixXd input)
{
    MatrixXd sig(input.rows(), input.cols());
    for (int i = 0; i < input.rows(); i++)
    {
        for (int j = 0; j < input.cols(); j++)
        {
            sig(i, j) = 1.0 / (1.0 + exp(-(input(i, j))));
        }
    }
    return sig;
}

//////////////////////////////////////////////////////////////

// same thing as above but now it is the derivative of the above (sigmoid) function
MatrixXd sigmoid_derivative(MatrixXd input)
{
    MatrixXd sig(input.rows(), input.cols());
    for (int i = 0; i < input.rows(); i++)
    {
        for (int j = 0; j < input.cols(); j++)
        {
            sig(i, j) = input(i, j) * (1 - input(i, j));
        }
    }
    return sig;
}

//////////////////////////////////////////////////////////////

// this is term by term multiplication of components of the matrices
MatrixXd term_by_term(MatrixXd lhs, MatrixXd rhs)
{
    MatrixXd ret(rhs.rows(), lhs.cols());
    for (int i = 0; i < lhs.rows(); i++)
    {
        for (int j = 0; j < rhs.cols(); j++)
        {
            ret(i, j) = lhs(i, j) * rhs(i, j);
        }
    }

    return ret;
}

//////////////////////////////////////////////////////////////

// neural network class
struct Neural_net
{
    /// @brief This neuralnet has 4 layers,
    // 1 input layer, 2 hidden layers, and 1 output layer
    int numberOfLayers = 4;

    // input layer will have 28x28 number of neurons
    // the 2 hidden layers have 16 neurons each
    // and the ouput layer has 10 neurons since we have 10 digits to predict
    int szOfLayers[4] = {784, 16, 16, 10};

    // delta
    vector<MatrixXd> weights, biases, delta_w, delta_b;
    double learning_rate;

    Neural_net(){};
    Neural_net(double alpha)
    {
        learning_rate = alpha;
        weights.resize(numberOfLayers - 1);
        biases.resize(numberOfLayers - 1);
        delta_w.resize(numberOfLayers - 1);
        delta_b.resize(numberOfLayers - 1);

        // assigning random weights and biases
        for (int i = 0; i < numberOfLayers - 1; i++)
        {

            weights[i] = randomNumberGenW(szOfLayers[i], szOfLayers[i + 1]);
            // cout << weights[i] << endl;
            biases[i] = randomNumberGenB(szOfLayers[i + 1]);
            // cout << biases[i] << endl;
            delta_w[i] = MatrixXd(szOfLayers[i], szOfLayers[i + 1]);
            delta_b[i] = MatrixXd(1, szOfLayers[i + 1]);
        }
    }

    // forward propagation - used only when testing not while training
    MatrixXd forwardProp(MatrixXd input)
    {
        for (int i = 0; i < numberOfLayers - 1; i++)
        {

            input = sigmoid(((input * weights[i]) + biases[i]));
        }
        return input;
    }

    void backProp(MatrixXd input, MatrixXd output)
    {
        vector<MatrixXd> layers;
        MatrixXd delta;
        layers.push_back(input);

        // forward prop
        for (int i = 0; i < numberOfLayers - 1; i++)
        {
            input = sigmoid(((input * weights[i]) + biases[i]));
            layers.push_back(input);
        }
        // after computing the output layer
        // we then compute the delta for the last layer
        // delta = outputlayer - actualoutput
        MatrixXd lastlayer = input;

        delta = term_by_term((input - output), sigmoid_derivative(layers[3]));
        delta_w[2] = delta_w[2] + ((layers[2].transpose()) * delta);
        delta_b[2] = delta_b[2] + (delta);
        // delta_w is the partial derivative of the loss/cost function with respect to w of a particular layer

        // MatrixXd lastlayer = input;
        // delta = term_by_term((lastlayer - output), (sigmoid_derivative(layers[numberOfLayers - 1])));
        // // std::cout << "row: " << delta_b[numberOfLayers - 2].rows() << "col: " << delta_b[numberOfLayers - 2].cols() << endl;
        // (delta_b[numberOfLayers - 2]) + delta;
        // // std::cout << "row: " << layers[numberOfLayers - 2].rows() << "col: " << layers[numberOfLayers - 2].cols() << endl;
        // (delta_w[numberOfLayers - 2]) + ((layers[numberOfLayers - 2]).transpose() * delta);

        for (int i = numberOfLayers - 3; i >= 0; i--)
        {
            delta = term_by_term((delta * (weights[i + 1]).transpose()), sigmoid_derivative(layers[i + 1]));

            delta_b[i] = delta_b[i] + (delta);
            delta_w[i] = delta_w[i] + ((layers[i].transpose()) * delta);
        }
    }

    void train(vector<MatrixXd> inputs, vector<MatrixXd> outputs)
    {
        // vector< vector <MatrixXd> > weightsAndBiases;

        for (int i = 0; i < 3; i++)
        {
            delta_b[i].setZero();
            delta_w[i].setZero();
        }

        for (int i = 0; i < (int)(inputs.size()); i++)
        {
            backProp(inputs[i], outputs[i]);
        }

        for (int i = 0; i < numberOfLayers - 1; i++)
        {
            delta_w[i] = delta_w[i] / (double)(inputs.size());
            weights[i] = weights[i] - (learning_rate * delta_w[i]);
            delta_b[i] = delta_b[i] / (double)(inputs.size());
            biases[i] = biases[i] - (learning_rate * delta_b[i]);
        }
        // weightsAndBiases.push_back(weights);
        // weightsAndBiases.push_back(biases);
        // return weightsAndBiases;
    }
};

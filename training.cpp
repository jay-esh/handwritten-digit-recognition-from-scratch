#include <stdio.h>
#include <fstream> // for file access
#include <iostream>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <vector>
#include <tuple>
#include <cmath>
#include "neuralnet.cpp"
// #include <json/writer.h>
// Json::Value event;

// #include "readMNIST.h"

const int MINI_BATCH_SIZE = 20;
vector<MatrixXd> inputData;
vector<MatrixXd> outputData;
Neural_net net = Neural_net(1.0);
int nCharacter, nCharacterTest;
character *Character, *CharacterTest;
vector<double> errorsRec;

void time_taken()
{
    std::cout << "Time: " << (int)(clock() * 1000.0 / CLOCKS_PER_SEC) << "ms" << std::endl;
}

void parseData()
{

    readMNIST("MNIST/train-images-idx3-ubyte", "MNIST/train-labels-idx1-ubyte",
              nCharacter, Character);

    std::cout << "Training Data is Loading :)" << std::endl;
    for (int i = 0; i < 60000; i++)
    {
        MatrixXd input(1, 784);
        MatrixXd output(1, 10);
        inputData.resize(60000);
        outputData.resize(60000);
        input = getIntValMatrix(i, Character);
        // printCharArray(input);
        inputData[i] = input;
        output = getLabelMx(i, Character);

        outputData[i] = output;
    }
    time_taken();
    std::cout << "**************************************" << std::endl;
}

void train()
{
    int epoch;
    vector<int> idx(60000);
    vector<MatrixXd> inputs(MINI_BATCH_SIZE), outputs(MINI_BATCH_SIZE);
    MatrixXd curr_output(1, 10);
    double error;
    double preverror;
    // vector < vector <MatrixXd> > weightsAndBiases;

    for (int i = 0; i < 60000; i++)
    {
        idx[i] = i;
    }

    for (epoch = 1; epoch <= 3; epoch++)
    {
        std::cout << "Epoch No: " << epoch << "started" << std::endl;
        std::random_shuffle(idx.begin(), idx.end());

        for (int k = 0; k < 60000; k += MINI_BATCH_SIZE)
        {
            // inputs.clear();
            // outputs.clear();

            for (int j = 0; j < MINI_BATCH_SIZE; j++)
            {
                inputs[j] = (inputData[idx[j + k]]);
                // printCharArray(inputData[idx[j + k]]);
                outputs[j] = (outputData[idx[j + k]]);
            }
            net.train(inputs, outputs);
        }
        // float errors[100];
        for (int l = 0; l < 60000; l++)
        {
            curr_output = net.forwardProp(inputData[l]);
            // cout << curr_output << endl;

            for (int m = 0; m < 10; m++)
            {
                error += (curr_output(0, m) - outputData[l](0, m)) * (curr_output(0, m) - outputData[l](0, m));
            }
        }
        // cout << net.forwardProp(inputData[10 - 1]) << endl;

        // vector<double> errorsRec;

        error /= 10.0;
        error /= 60000.0;

        // if (error > preverror)
        // {
        //     cout << "up" << endl;
        // }
        // if (error < preverror)
        // {
        //     cout << "down" << endl;
        // }
        errorsRec.push_back(error);
        preverror = error;
        std::cout
            << "Error: " << error << std::endl;

        std::cout
            << "Epoch No: " << epoch << "finished" << std::endl;

        time_taken();
        std::cout << std::endl;
        // cout << net.forwardProp()
    }
}

void test()
{
    // int nCharacterTest;
    // character *CharacterTest;

    readMNIST("MNIST/t10k-images-idx3-ubyte", "MNIST/t10k-labels-idx1-ubyte",
              nCharacterTest, CharacterTest);

    for (int i = 0; i < 20; i++)
    {
        MatrixXd newInput;
        MatrixXd output;
        // int sampleNo = rand01() * 10;
        std::cout << "Label " << LabelToText(CharacterTest[i].Label) << "\n";
        newInput = getIntValMatrix(i, CharacterTest);
        // printCharArray(newInput);
        // inputData[i] = input;
        printImage(i, nCharacterTest, CharacterTest);
        output = getLabelMx(i, CharacterTest);
        // cout << output << endl;
        // outputData[i] = output;
        MatrixXd neural_net_output = net.forwardProp(newInput);
        for (int i = 0; i < neural_net_output.cols(); i++)
        {
            if (neural_net_output(0, i) >= (double)(0.5))
            {
                std::cout << "Prediction: " << i << std::endl;
                std::cout << "Probability: " << neural_net_output(0, i) << std::endl
                          << std::endl;
            }
        }
    }
}

int main()
{
    parseData();
    train();
    // weightsnbias.push_back(errorsRec);
    test();
    return 0;
}

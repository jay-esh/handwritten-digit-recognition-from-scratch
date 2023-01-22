#include <stdio.h>
#include <fstream> // for file access
#include <iostream>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <vector>
#include <tuple>
#include <cmath>

using namespace std;

class Matrix
{
public:
    int row;
    int col;
    double **array;

    Matrix() : row(0), col(0), array(nullptr){};

    Matrix(int rows, int cols)
    {
        row = rows;
        col = cols;

        array = new double *[row];
        for (int i = 0; i < row; i++)
        {
            array[i] = new double[col];
            for (int j = 0; j < col; j++)
            {
                array[i][j] = 0.0;
            }
        }
    };

    void display()
    {
        cout << "[ " << endl;
        for (int i = 0; i < row; i++)
        {
            cout << " [ ";
            for (int j = 0; j < col; j++)
            {

                cout << array[i][j] << ", ";
            }
            cout << "]" << endl;
        }

        cout << "]" << endl;
    }
};

Matrix set(Matrix a, int input)
{
    for (int i = 0; i < a.row; i++)
    {
        for (int j = 0; j < a.col; j++)
        {
            a.array[i][j] = (double)input;
        }
    }

    return a;
}

Matrix scalar_add_sub_mult(Matrix a, Matrix b, char c)
{
    Matrix ret = Matrix();
    if ((a.row != b.row) || (a.col != b.col))
    {
        cout << "Matrix addition cannot be executed on matrices of different sizes";
        return ret;
    }

    Matrix newret = Matrix(a.row, a.col);
    for (int i = 0; i < a.row; i++)
    {
        for (int j = 0; j < a.col; j++)
        {
            if (c == 'a')
            {
                newret.array[i][j] = a.array[i][j] + b.array[i][j];
            }
            else if (c == 's')
            {
                newret.array[i][j] = a.array[i][j] - b.array[i][j];
            }
            else
            {
                newret.array[i][j] = a.array[i][j] * b.array[i][j];
            }
        }
    }

    return newret;
}

double dot_prod(Matrix a, Matrix b)
{
    if ((a.col != 1) || (b.col != 1) || (a.row != b.row))
    {
        cout << "dot product error!";
        return 0.0;
    }

    double ret = 0.0;
    for (int i = 0; i < a.row; i++)
    {
        ret += a.array[i][0] * b.array[i][0];
    }

    return ret;
}

Matrix mx_mult(Matrix a, Matrix b)
{
    if (a.col != b.row)
    {
        Matrix ret = Matrix();
        cout << "matrix multiplication error!!\n";
        return ret;
    }

    Matrix ret = Matrix(a.row, b.col);
    int itr = 0;

    for (int i = 0; i < a.row; i++)
    {
        for (int j = 0; j < b.col; j++)
        {
            double out = 0.0;
            for (int k = 0; k < b.row; k++)
            {
                out += a.array[i][k] * b.array[k][i];
            }
            ret.array[i][j] = out;
        }
    }

    return ret;
}

Matrix transpose(Matrix a)
{
    Matrix ret = Matrix(a.col, a.row);
    for (int i = 0; i < a.row; i++)
    {
        for (int j = 0; j < a.col; j++)
        {
            ret.array[j][i] = a.array[i][j];
        }
    }

    return ret;
}

double sigmoid(double x)
{
    return (1.0 / (1.0 + exp(-x)));
}

// here we consider x to be a sigmoid function
double sigmoid_derivative(double x)
{
    return (sigmoid(x) * (1 + sigmoid(x)));
}

Matrix sigmoid_mx(Matrix mx)
{
    Matrix ret = Matrix(mx.row, mx.col);
    for (int i = 0; i < mx.row; i++)
    {
        for (int j = 0; j < mx.col; j++)
        {
            ret.array[i][j] = sigmoid(mx.array[i][j]);
        }
    }
    return ret;
}

Matrix sigmoid_deriv_mx(Matrix mx)
{
    Matrix ret = Matrix(mx.row, mx.col);
    for (int i = 0; i < mx.row; i++)
    {
        for (int j = 0; j < mx.col; j++)
        {
            ret.array[i][j] = sigmoid_derivative(mx.array[i][j]);
        }
    }
    return ret;
}

// int main(void)
// {
//     Matrix a = Matrix(10, 2);
//     Matrix b = Matrix(2, 10);
//     // a.display();
//     b = set(b, 4124);
//     // a = set(a, 1124124);
//     // a.display();
//     b.display();
//     cout << sigmoid(100) << endl;
//     Matrix ret = transpose(b);
//     // Matrix ret = scalar_add_sub(a, b, 's');
//     // Matrix ret = mx_mult(a, b);
//     ret.display();
//     // cout << dot_prod(a, b) << endl;
//     return 0;
// }

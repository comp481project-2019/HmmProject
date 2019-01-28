#include <iostream>         // for IO
#include <vector>           // for 2D vector
#include <stdlib.h>         // srand, rand 
#include <time.h>           // time for generating random number
#include <chrono>           // for timing execution of matrix multiply
#include <fstream>          // for writing timing results to csv
#include <bits/stdc++.h>    // for sorting vectors

using namespace std;
using namespace std::chrono;




/*
    Prints an input matrix to standard output one row at a time.

    Params:
        IntMatrix matrix: a matrix to be printed

    Return: 
        void: nothing to return because it is simply printing
*/
void print_double_matrix(vector<vector<double> > &matrix)
{
    int rows = matrix.size();
    int cols = matrix[0].size();

    int i, j;

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            cout << matrix[i][j] << ", ";
        }
        cout << "\n";
    }
    cout << "\n";
}

void print_int_vector(vector<int> vec)
{
    for (int i = 0; i < vec.size(); i++)
    {
        cout << vec[i] << ", ";
    }
    cout << "\n";
}

vector<vector<double> > forward(vector<vector<double> > &transition, vector<vector<double> > &emission,  vector<int> &observations, int stop)
{
    int numCols = stop + 1; //observations.size() + 1;
    int numRows = transition.size();
    int row, col, p;
    double stateProb, result;

    vector<vector<double> > resultMatrix(numRows, vector<double>(numCols)); // Defaults to zero initial value


    resultMatrix[0][0] = 1.0;

    for (col = 1; col < numCols; col++)
    {
        for (row = 1; row < numRows; row++)
        {
            for (p = 0; p < numRows; p++)
            {
                stateProb += transition[p][row]*resultMatrix[p][col-1];
            }
            resultMatrix[row][col] = stateProb*emission[row-1][observations[col-1]];
            stateProb = 0;
        }
    }

    return resultMatrix;
    
}


void backward(vector<vector<double> > transition, vector<vector<double> > emission,  vector<int> observations, int stop)
{
    int numCols = observations.size() + 1 - stop;
    int numRows = transition.size();
    int row, col, p;
    double stateProb, result;

    vector<vector<double> > resultMatrix(numRows, vector<double>(numCols)); // Defaults to zero initial value
    

    for (row = 1; row < numRows; row++)
    {
        resultMatrix[row][numCols-1] = 1;
    }
    print_double_matrix(resultMatrix);
    
    for (col = numCols-2; col >= stop; col--)
    {
        for (row = 1; row < numRows; row++)
        {
            for (p = 1; p < numRows; p++)
            {
                resultMatrix[row][col] += transition[row][p]*emission[p-1][observations[col]]*resultMatrix[p][col+1];
            }
            
        }
    }

    print_double_matrix(resultMatrix);
    
}


/*
    

    params:
        None

    return:
        int: 0 if program successfully executes
*/
int main()
{
    vector<vector<double> > transition { { 0, 0.7, 0.3 },
                                    { 0, 0.6, 0.4 },
                                    {0, 0.3, 0.7}};

    vector<vector<double> > emission { { 0.2, 0.8 },
                                    { 0.6, 0.4 }};

                                
    vector<int> observations {0,1, 1,1,0,0};

    print_double_matrix(transition);
    print_double_matrix(emission);

    vector<vector<double> > result = forward(transition, emission, observations, 6);
    cout << "forward\n";
    print_double_matrix(result);
    cout << "backward\n";
    backward(transition, emission, observations, 0);

	return 0;
}

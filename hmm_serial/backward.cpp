#include <iostream>         // for IO
#include <vector>           // for 2D vector
#include <stdlib.h>         // srand, rand 
#include <time.h>           // time for generating random number
#include <chrono>           // for timing execution of matrix multiply
#include <fstream>          // for writing timing results to csv
#include <bits/stdc++.h>    // for sorting vectors
#include <math.h>    

using namespace std;
using namespace std::chrono;





/*
    Prints an input matrix to standard output one row at a time.

    Params:
        vector<vector<float> > matrix: a matrix to be printed

    Return: 
        void: nothing to return because it is simply printing
*/
void print_float_matrix(vector<vector<float> > &matrix)
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

/*
    print all the values in a vector of ints
*/
void print_int_vector(vector<int> &vec)
{
    for (int i = 0; i < vec.size(); i++)
    {
        cout << vec[i] << ", ";
    }
    cout << "\n";
}

/*
    print all the values in a vector of floats
*/
void print_float_vector(vector<float> &vec)
{
    for (int i = 0; i < vec.size(); i++)
    {
        cout << vec[i] << ", ";
    }
    cout << "\n";
}

/*
    given a matrix, log each value to get them in log space
*/ 
void convertMatrixToLogSpace(vector<vector<float> > &probs)
{   
    int i, j;
    int row = probs.size();
    int col = probs[0].size();

    for (i = 0; i < row; i++)
    {
        for (j = 0; j < col; j++)
        {
            probs[i][j] = log(probs[i][j]);
        }
    }
}


/*
    given a vector, log each value to get them in log space
*/ 
void convertVectorToLogSpace(vector<float> &probs)
{   
    int row = probs.size();
    int i;
    for (i = 0; i < row; i++)
    {
        
        probs[i] = log(probs[i]);
        
    }
}

/*
    given an array of loged values perform the operation exp(probs[i][j]) in order to get the actual value
*/
void convertMatrixFromLogSpace(vector<vector<float> > &probs)
{   
    int i, j;
    int row = probs.size();
    int col = probs[0].size();

    for (i = 0; i < row; i++)
    {
        for (j = 0; j < col; j++)
        {
            probs[i][j] = exp(probs[i][j]);
        }
    }
}

/*
    performs the equivalent operation to log(x+y) but with log(x) and log(y)
*/
float logsum(float x, float y)
{
    if (x == 0 || y == 0)
        return x+y;
    return min(x,y) + log(1+exp(abs(y-x)));
} 

/*
    perform the backward algorithm on HMM parameters
*/
vector<vector<float> > backward(vector<vector<float> > &transition, vector<vector<float> > &emission,  vector<int> &observations, vector<float> &pi, int stop)
{
    int numCols = observations.size() - stop + 1;
    int numRows = transition.size();
    int row, col, p;
    float stateProb, result;

    vector<vector<float> > resultMatrix(numRows, vector<float>(numCols)); // Defaults to zero initial value
    

    for (row = 0; row < numRows; row++)
    {
        resultMatrix[row][numCols-1] = 0;
    }

    
    for (col = numCols-2; col >= 0; col--)
    {
        for (row = 0; row < numRows; row++)
        {
            for (p = 0; p < numRows; p++)
            {
                resultMatrix[row][col] = logsum(transition[row][p]+emission[p][observations[col]]+resultMatrix[p][col+1], resultMatrix[row][col]);
            }
            
        }
    }

    if (stop == 0)
    {
        // probabilities of returning to inital state. the sum should equal forward
        for (row = 0; row < numRows; row++)
        {
            resultMatrix[row][0] = pi[row] + resultMatrix[row][1] + emission[row][observations[0]];
        }
    }

    return resultMatrix;
    
}

/*
    params:
        None

    return:
        int: 0 if program successfully executes
*/
int main()
{ 

    vector<float> pi {0.8, 0.2};
    vector<vector<float> > transition {{ 0.6, 0.4 },
                                    {0.3, 0.7}};

    vector<vector<float> > emission { { 0.3, 0.4, 0.3 },
                                    { 0.4, 0.3, 0.3 }};

                                
    vector<int> observations {0,1,2,2};

    print_float_matrix(transition);
    print_float_matrix(emission);

    convertMatrixToLogSpace(transition);
    convertMatrixToLogSpace(emission);
    convertVectorToLogSpace(pi);

    print_float_matrix(transition);
    print_float_matrix(emission);
    print_float_vector(pi);



    vector<vector<float> > result = backward(transition, emission, observations, pi, 0);
    cout << "backward\n";
    print_float_matrix(result);

    convertMatrixFromLogSpace(result);

    print_float_matrix(result);

	return 0;
}
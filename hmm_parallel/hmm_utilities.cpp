#include <iostream>      
#include <vector>           
#include <stdlib.h>         
#include <iostream>      
#include <vector>           
#include <cstdlib>        
#include <time.h>          
#include <omp.h>           
#include <fstream> 
#include <cmath>  
#include "hmm_utilities.h"          
     

using namespace std;

struct HmmParams {
    vector<vector<double> > transition;
    vector<vector<double> > emission;
    vector<double> initial;
};


struct HmmData {
    int numHmmInputs = 0;
    vector<HmmParams> hmmParamList;
    vector<vector<vector<int> > > trainingSets;
};

/*
    given a matrix, log each value to get them in log space
*/ 
void convert_matrix_to_log_space(vector<vector<double> > &probs)
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
void convert_vector_to_log_space(vector<double> &probs)
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
void convert_matrix_from_log_space(vector<vector<double> > &probs)
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
    given a vector, log each value to get them in log space
*/ 
void convert_vector_from_log_space(vector<double> &probs)
{   
    int row = probs.size();
    int i;
    for (i = 0; i < row; i++)
    {
        
        probs[i] = exp(probs[i]);
        
    }
}

/*
    performs the equivalent operation to log(x+y) but with log(x) and log(y)
*/
double logsum(double x, double y)
{
    if (x == 0 || y == 0)
        return x+y;
    return min(x,y) + log(1+exp(abs(y-x)));
} 

/*
    Prints an input matrix to standard output one row on each line with 
    values in a given row seperated by comma.

    Params:
        vector<vector<double> > matrix: a matrix to be printed

    Return: 
        void: nothing to return because it is simply printing
*/
void print_double_matrix(vector<vector<double> > &matrix)
{
    int rows = matrix.size();
    int cols;

    int i, j;

    for (i = 0; i < rows; i++)
    {
        cols = matrix[i].size();
        for (j = 0; j < cols; j++)
        {
            cout << matrix[i][j] << ", ";
        }
        cout << "\n";
    }
    cout << "\n";
}

/*
    Prints an input matrix to standard output one row on each line with 
    values in a given row seperated by comma.

    Params:
        vector<vector<int> > matrix: a matrix to be printed

    Return: 
        void: nothing to return because it is simply printing
*/
void print_int_matrix(vector<vector<int> > &matrix)
{
    int rows = matrix.size();
    int cols;

    int i, j;

    for (i = 0; i < rows; i++)
    {
        cols = matrix[i].size();
        for (j = 0; j < cols; j++)
        {
            cout << matrix[i][j] << ", ";
        }
        cout << "\n";
    }
    cout << "\n";
}

/*
    prints values in a vector of ints seperated by comma

    params: 
        vector<int> vec: vector to print
*/
void print_int_vector(vector<int> vec)
{
    for (int i = 0; i < vec.size(); i++)
    {
        cout << vec[i] << ", ";
    }
    cout << "\n";
}

/*
    prints values in a vector of doubles seperated by comma

    params: 
        vector<double> vec: vector to print
*/
void print_double_vector(vector<double> vec)
{
    for (int i = 0; i < vec.size(); i++)
    {
        cout << vec[i] << ", ";
    }
    cout << "\n\n";
}

/*
    Builds a randomized probability matrix of given dimension. Each row of the matrix sums to 1

    Params:
        int numRows: number of rows in the matrix
        int numCols: numberof cols in the matrix

    Return: 
        vector<vector<double> >: a probability matrix
*/
vector<vector<double> > generate_probability_matrix(int numRows, int numCols)
{

    vector<vector<double> > matrix;
    vector<double> rowSum(numRows);
    int i, j;
    double random;
    for (i = 0; i < numRows; i++)
    {
        vector<double> col;
        for (j = 0; j < numCols; j++)
        {
            random = 1 + (float)rand()/(RAND_MAX/(10 - 1)); // ensures the random number is in the input range
            col.push_back(random);
        }
        matrix.push_back(col);
    }


    // sum of values in each row
    for (int row = 0; row < numRows; row++)
    {
        for (int col = 0; col < numCols; col++)
        {
            rowSum[row] += matrix[row][col];
        }
    }

     // normalize each row so its values usm to 1
    for (int row = 0; row < numRows; row++)
    {
        for (int col = 0; col < numCols; col++)
        {
            matrix[row][col] = matrix[row][col]/rowSum[row];
        }
    }

    return matrix;

}

/*
    Builds a probability matrix of given dimension such that all probabilites are the same,
    and all rows sum to 1;

    Params:
        int numRows: number of rows in the matrix
        int numCols: numberof cols in the matrix

    Return: 
        vector<vector<double> >: a probability matrix
*/
vector<vector<double> > generate_equal_probability_matrix(int numRows, int numCols)
{

    vector<vector<double> > matrix;
    int i, j;
    double probs = 1.0/numCols;
    for (i = 0; i < numRows; i++)
    {
        vector<double> col;
        for (j = 0; j < numCols; j++)
        {
            col.push_back(probs);
        }
        matrix.push_back(col);
    }

    return matrix;

}


/*
    Builds a randomized probability vector such that its values sum to one

    Params:
        int numProbs: the number of probabilties which summed equals 1

    Return: 
        vector<double>: a probability vector
*/
vector<double> generate_probability_vector(int numProbs)
{

    vector<double> vector;
    double rowSum;
    int i;
    double random;

    for (i = 0; i < numProbs; i++)
    {
        random = 1 + static_cast<float>(rand()) /(static_cast<float>(RAND_MAX/(10 - 1))); // ensures the random number is in the input range
        vector.push_back(random);
    }

    // sum all the values
    for (int s = 0; s < numProbs; s++)
    {
        rowSum += vector[s];
    }

    // divide each value by the sum of values to normalize the data
    for (int s = 0; s < numProbs; s++)
    {
        vector[s] = vector[s]/rowSum;
    }

    return vector;

}


/*
    Builds a probability vector such that its values sum to one and all probabiities are equal

    Params:
        int numProbs: the number of probabilties which summed equals 1

    Return: 
        vector<double>: a probability vector
*/
vector<double> generate_equal_probability_vector(int numProbs)
{

    vector<double> vector;
    int i;
    double probs = 1.0/numProbs;

    for (i = 0; i < numProbs; i++)
    {
        vector.push_back(probs);
    }

    return vector;

}

/*
    Builds a randomized vector of observations, given a vector size and number of possible emissions

    Params:
        int numEmissions: number of possible emissions
        int size: size of observation vector

    Return: 
        vector<int>: a vector of randomly generates values between 0 and numEmissions - 1 inclusive
*/
vector<int> generate_observation_vector(int numEmissions, int size)
{

    vector<int> vector;
    int i;

    for (i = 0; i < size; i++)
    {
        vector.push_back(rand() % (numEmissions));
    }

    return vector;
}


/*
    Builds a randomized training set of observations for an hmm. 
    
    Note: randomized training data is only useful for testing runtimes of randomly generated models, 
    not for obtaining useful models.

    Params:
        int numEmissions: The number of possible emissions in the model 
        vector<int> sizes: a vector of integers representing the size of each random training set 

    Return: 
        vector<vector<int> >: a 2d vector where each row is a random observation sequence
*/
vector<vector<int> > generate_training_set(int numEmissions, vector<int> sizes)
{

    vector<vector<int> > trainingSet;
    int i;

    for (i = 0; i < sizes.size(); i++)
    {
        trainingSet.push_back(generate_observation_vector(numEmissions, sizes[i]));
    }

    return trainingSet;
}

/*  
    Reads input data from a specially formatted hmm input file. The program user must input the name of the file to be parsed. 

    How to read the input file:
        1. first line is integer (N) representing the number of hmm models that exist in the file
        2. following line 1 are N blocks, each representing a different hmm model to test
            a. line 1 of a block is the number of states (S) in the model
            b. line 2 of the block is the number of possible emissions (E) from the model
            c. the next S lines contain S space-separated floating point numbers representing transition probabilities
            d. the next S lines contain E space-separated floating point numbers representing emission probabilities
            e. the following line contains the number of observation sequences (T) to be used for training the model
            f. following this line there are O two line training data blocks
                i. the first line of a training data block indicates the number of observations (O) in the observation sequence
                ii. the second line contains O space seperated integers representing the different observations

    return:
        HmmData: an object representing data contained within a file for numerous HMMs 

*/
HmmData read_file_data()
{
    ifstream fin;
    string inputFileName;
    // cout << "Enter name of input file:\n";
    // cin >> inputFileName;
    fin.open("hmmInput.txt");

    

    HmmData fData;
    int numStates, numEmissions, numObservationSets, i, j;
    fin >> fData.numHmmInputs;

    for (int hmmIndex = 0; hmmIndex < fData.numHmmInputs; hmmIndex++)
    {

        fin >>  numStates >> numEmissions;

        vector<double> initial(numStates);
        vector<vector<double> > transition(numStates, vector<double>(numStates));
        vector<vector<double> > emission(numStates, vector<double>(numEmissions));
        vector<vector<int> > training;
        for (i = 0; i < numStates; i++)
        {
            fin >> initial[i];
        }

        for (i = 0; i < numStates; i++)
        {
            for (j = 0; j < numStates; j++)
            {
                fin >> transition[i][j];
            }
        }

        for (i = 0; i < numStates; i++)
        {
            for (j = 0; j < numEmissions; j++)
            {
                fin >> emission[i][j];
            }
        }

        int numObservations, obs;
        fin >> numObservationSets;
        vector<int> observations;
        for (i = 0; i < numObservationSets; i++)
        {
            fin >> numObservations;
            for (j = 0; j < numObservations; j++)
            {
                fin >> obs;
                observations.push_back(obs);
            }
            training.push_back(observations);
            observations.clear();
        }

        HmmParams params;
        params.initial = initial;
        params.transition = transition;
        params.emission = emission;
    
        fData.hmmParamList.push_back(params);
        fData.trainingSets.push_back(training);

    }
    
    fin.close();

    return fData;
}

/*
    given a number of states and number of emissions, generate a random set of parameters for a HMM

    params: 
        int numStates: the number of states in the random hidden markov model
        int numEmissions: the number of emissions in the random hidden markov model
    return:
        HmmParams: the paramaters for the random HMM
*/
HmmParams generate_random_hmm(int numStates, int numEmissions)
{
    HmmParams params;
    params.initial = generate_probability_vector(numStates);
    params.transition = generate_probability_matrix(numStates, numStates);
    params.emission = generate_probability_matrix(numStates, numEmissions);


    return params;
}
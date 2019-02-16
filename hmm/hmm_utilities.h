#ifndef HMM_UTIL
#define HMM_UTIL
#include <iostream>      
#include <vector>           
#include <stdlib.h> 
#include "hmm_utilities.cpp"                 

using namespace std;


/*
    Prints an input matrix to standard output one row on each line with 
    values in a given row seperated by comma.

    Params:
        vector<vector<double> > matrix: a matrix to be printed

    Return: 
        void: nothing to return because it is simply printing
*/
void print_double_matrix(vector<vector<double> > &matrix);


/*
    performs the equivalent operation to log(x+y) but with log(x) and log(y)
*/
double logsum(double x, double y);

/*
    Prints an input matrix to standard output one row on each line with 
    values in a given row seperated by comma.

    Params:
        vector<vector<double> > matrix: a matrix to be printed

    Return: 
        void: nothing to return because it is simply printing
*/
void print_double_matrix(vector<vector<double> > &matrix);

/*
    Prints an input matrix to standard output one row on each line with 
    values in a given row seperated by comma.

    Params:
        vector<vector<int> > matrix: a matrix to be printed

    Return: 
        void: nothing to return because it is simply printing
*/
void print_int_matrix(vector<vector<int> > &matrix);

/*
    prints values in a vector of ints seperated by comma

    params: 
        vector<int> vec: vector to print
*/
void print_int_vector(vector<int> vec);

/*
    prints values in a vector of doubles seperated by comma

    params: 
        vector<double> vec: vector to print
*/
void print_double_vector(vector<double> vec);

/*
    Builds a randomized probability matrix of given dimension. Each row of the matrix sums to 1

    Params:
        int numRows: number of rows in the matrix
        int numCols: numberof cols in the matrix

    Return: 
        vector<vector<double> >: a probability matrix
*/
vector<vector<double> > generate_probability_matrix(int numRows, int numCols);

/*
    Builds a probability matrix of given dimension such that all probabilites are the same,
    and all rows sum to 1;

    Params:
        int numRows: number of rows in the matrix
        int numCols: numberof cols in the matrix

    Return: 
        vector<vector<double> >: a probability matrix
*/
vector<vector<double> > generate_equal_probability_matrix(int numRows, int numCols);


/*
    Builds a randomized probability vector such that its values sum to one

    Params:
        int numProbs: the number of probabilties which summed equals 1

    Return: 
        vector<double>: a probability vector
*/
vector<double> generate_probability_vector(int numProbs);


/*
    Builds a probability vector such that its values sum to one and all probabiities are equal

    Params:
        int numProbs: the number of probabilties which summed equals 1

    Return: 
        vector<double>: a probability vector
*/
vector<double> generate_equal_probability_vector(int numProbs);

/*
    Builds a randomized vector of observations, given a vector size and number of possible emissions

    Params:
        int numEmissions: number of possible emissions
        int size: size of observation vector

    Return: 
        vector<int>: a vector of randomly generates values between 0 and numEmissions - 1 inclusive
*/
vector<int> generate_observation_vector(int numEmissions, int size);


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
vector<vector<int> > generate_training_set(int numEmissions, vector<int> sizes);

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
HmmData read_file_data();


/*
    given a number of states and number of emissions, generate a random set of parameters for a HMM

    params: 
        int numStates: the number of states in the random hidden markov model
        int numEmissions: the number of emissions in the random hidden markov model
    return:
        HmmParams: the paramaters for the random HMM
*/
HmmParams generate_random_hmm(int numStates, int numEmissions);

#endif
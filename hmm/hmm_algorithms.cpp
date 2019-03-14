#include <iostream>      
#include <vector>           
#include <stdlib.h>         
#include <time.h>          
#include <omp.h>           
#include <fstream> 
#include <cmath>  
#include "hmm_utilities.h" 
#include "hmm_algorithms.h"


using namespace std;



/*
    given a matrix, log each value to get them in log space
    params:
        vector<vector<double> > &probs: a 2d vector of probabilities
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
    params:
        vector<vector<double> > &probs: a vector of probabilities
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
    params:
        vector<vector<double> > &probs: a 2d vector of probabilities
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
    params:
        vector<vector<double> > &probs: a vector of probabilities
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
    The HMM forward algorithm uses dynamic programming to estimate the probablity of being in state i at time t,
    having seen a particular sequence of observations. Summing the probabilites of being in each state when the final 
    observation occurs at time T, gives the probability of seeing the entire observation sequence. 

    params:
        vector<vector<double> > &transition: a matrix representing the transition probabilities of the HMM
        vector<vector<double> > &emission: a matrix representing the probability of making an observation from a given state
        vector<double> &pi: a vector representing the probability of starting in a given state
        vector<int> &observations: a vector representing a sequence of observations
        int stop: an integer indicating at what time to stop in the sequence of observations. Generally this will equal the
        number of observations, however, if only interested in the probability up to a point, this argument prevents the need to 
        bild the entire forward matrix.
    return:
        vector<vector<double> >: a forward probability matrix that represent the probability of being in state i at time t
        having already seen observations 0,1,...,t (one observation at each time t).

*/
vector<vector<double> > forward_serial_log (vector<vector<double> > &transition, vector<vector<double> > &emission,  vector<double> &pi, vector<int> &observations, int stop)
{
    int numCols = stop;
    int numRows = transition.size();
    double stateProb;

    vector<vector<double> > resultMatrix(numRows, vector<double>(numCols)); // Defaults to zero initial value

    // initialization
    for (int row = 0; row < numRows; row++){
        resultMatrix[row][0] = pi[row] + emission[row][observations[0]];
    }

    // calculate forward probabilities
    for (int col = 1; col < numCols; col++)
    {
        for (int row = 0; row < numRows; row++)
        {
            for (int p = 0; p < numRows; p++)
            {
                stateProb = transition[p][row] + resultMatrix[p][col-1] + emission[row][observations[col]];
                resultMatrix[row][col] = logsum(stateProb, resultMatrix[row][col]);
            }
        }
    }

    return resultMatrix;
    
}

/*
    The HMM backward algorithm uses dynamic programming to estimate the probablity of being in state i at time t, and
    seeing a particular sequence of observations after transitioning to the next state at time t + 1

    params:
        vector<vector<double> > &transition: a matrix representing the transition probabilities of the HMM
        vector<vector<double> > &emission: a matrix representing the probability of making an observation from a given state
        vector<double> &pi: a vector representing the probability of starting in a given state
        vector<int> &observations: a vector representing a sequence of observations
    return:
        vector<vector<double> >: a backward probability matrix that represent the probability of being in state i at time t
        and then seeing observations t+1,t+2,...,T, where T is the total number of observations (one observation at each time t)

*/
vector<vector<double> > backward_serial_log(vector<vector<double> > &transition, vector<vector<double> > &emission, vector<double> &pi, vector<int> &observations)
{

    int numCols = observations.size();
    int numRows = transition.size();
    double stateProb, result;
    
    vector<vector<double> > resultMatrix(numRows, vector<double>(numCols)); // Defaults to zero initial value
    
    // initial probabilities
    for (int row = 0; row < numRows; row++)
    {
        resultMatrix[row][numCols-1] = 1;
    }
    
    // calculate backward probabilities 
    for (int col = numCols-2; col >= 0; col--)
    {
        for (int row = 0; row < numRows; row++)
        {
            for (int p = 0; p < numRows; p++)
            {
                stateProb = transition[row][p] + emission[p][observations[col+1]] + resultMatrix[p][col+1];
                resultMatrix[row][col]  = logsum(stateProb, resultMatrix[row][col]);
            }
            
        }
    }
    
    return resultMatrix;
    
}


/*
    Using the forward-backward algorithm, predict HMM parameters that best estimate a training set of 
    observations.

    params:
        HmmParams &params: starting parameters of an HMM to be trained
        vector<vector<int> > &training: a set of observations sequences with which to train the HMM
        int iterations: the number of iterations of training to go through

    return:
        HmmParams: a new set of HMM parameters that represent an improved estimate given the training data

*/
HmmParams baum_welch_serial(HmmParams &params, vector<vector<int> > &training, int iterations)
{
    int it;

    vector<vector<double> > transition = params.transition;
    vector<vector<double> > emission = params.emission;
    vector<double> initial = params.initial;

    convert_matrix_to_log_space(transition);
    convert_matrix_to_log_space(emission);
    convert_vector_to_log_space(initial);
    
    int numStates = initial.size();
    int numObservs;

    // this represent the probability of an observation sequence to be trained on.
    double za;


    for (it = 0; it < iterations; it++)
    {

        vector<vector<double> > transitionUpdate(transition.size(), vector<double>(transition[0].size()));
        vector<vector<double> > emissionUpdate(emission.size(), vector<double>(emission[0].size()));
        vector<double> initialUpdate(initial.size());   
        
        for (int o = 0; o < training.size(); o++){
            vector<int> observations = training[o];
            vector<vector<double> > alpha;
            vector<vector<double> > beta;
            numObservs = observations.size();

           
            alpha = forward_serial_log(transition, emission, initial, observations, observations.size());

        
            beta = backward_serial_log(transition, emission, initial, observations);

            

            // given alpha, calculate the probability of seeing the observation sequence
            za = 0;
            for (int i = 0; i < alpha.size(); i++)
            {
                za = logsum(za, alpha[i][alpha[0].size()-1]);
            }
            
             
            // update initial
            for (int row = 0; row < numStates; row++)
            {
                initialUpdate[row] = logsum(alpha[row][0] + beta[row][0] - za, initialUpdate[row]);
            }

              
            // update emission probs
            for (int col = 0; col < numObservs; col++)
            {
                for (int row = 0; row < numStates; row++)
                {
                    
                    emissionUpdate[row][observations[col]] = logsum(alpha[row][col] + beta[row][col] - za, emissionUpdate[row][observations[col]]); 
                }
            
            }
                
            // update transition probs
            for (int i = 1; i < numObservs; i++)
            {
                for (int s1 = 0; s1 < numStates; s1++)
                {
                    for (int s2 = 0; s2 < numStates; s2++)
                    {
                        transitionUpdate[s1][s2] = logsum(alpha[s1][i-1] + transition[s1][s2] + emission[s2][observations[i]] + beta[s2][i] - za, transitionUpdate[s1][s2]);
                    }
                }
            }
                
        }
        

        // create denominators for normalization
        double initialSum = 0;
        vector<double> transitionSum(numStates);
        vector<double> emissionSum(emission.size());

       
        // sum of initals
        for (int s = 0; s < numStates; s++)
        {
            initialSum = logsum(initialUpdate[s], initialSum);
        }
    
    
        // sum of values in each tranistion row
        for (int row = 0; row < numStates; row++)
        {
            for (int col = 0; col < numStates; col++)
            {
                transitionSum[row] = logsum(transitionSum[row], transitionUpdate[row][col]);
            }
        }
    

    
        //sum of values in each emssion row
        for (int row = 0; row < numStates; row++)
        {
            for (int col = 0; col < emissionUpdate[0].size(); col++)
            {
                emissionSum[row] = logsum(emissionSum[row], emissionUpdate[row][col]);
            }
        }
            
        

        // normalization step:

        // 1. normailze initial
        for (int s = 0; s < numStates; s++)
        {
            initial[s] = initialUpdate[s] - initialSum;
        }

        // 2. normalize transition
        for (int row = 0; row < numStates; row++)
        {
            for (int col = 0; col < numStates; col++)
            {
                transition[row][col] = transitionUpdate[row][col] - transitionSum[row];
            }
        }

        // 3. normalize emission
        for (int row = 0; row < numStates; row++)
        {
            for (int col = 0; col < emissionUpdate[0].size(); col++)
            {
                emission[row][col] = emissionUpdate[row][col] - emissionSum[row];
            }
        }        

    }

    HmmParams newParams;

    convert_vector_from_log_space(initial);
    convert_matrix_from_log_space(transition);
    convert_matrix_from_log_space(emission);
    newParams.initial = initial;
    newParams.transition = transition;
    newParams.emission = emission;

    return newParams;

}


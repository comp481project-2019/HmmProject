#ifndef HMM_ALGO
#define HMM_ALGO
#include <iostream>      
#include <vector>           
#include <stdlib.h> 
#include "hmm_algorithms_para.cpp"                 

using namespace std;

/*
    given a matrix, log each value to get them in log space
    params:
        int num_threads: number of threads with which to run in parallel
*/ 
void convert_matrix_to_log_space_para(vector<vector<double> > &probs, int num_threads);


/*
    given a vector, log each value to get them in log space
    params:
        int num_threads: number of threads with which to run in parallel
*/ 
void convert_vector_to_log_space_para(vector<double> &probs, int num_threads);


/*
    given an array of loged values perform the operation exp(probs[i][j]) in order to get the actual 
    params:
        int num_threads: number of threads with which to run in parallel
*/
void convert_matrix_from_log_space_para(vector<vector<double> > &probs, int num_threads);

/*
    given a vector, log each value to get them in log space
    params:
        int num_threads: number of threads with which to run in parallel
*/ 
void convert_vector_from_log_space_para(vector<double> &probs, int num_threads);

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
        int num_threads: number of threads to run in parallel
    return:
        vector<vector<double> >: a forward probability matrix that represent the probability of being in state i at time t
        having already seen observations 0,1,...,t (one observation at each time t).

*/
vector<vector<double> > forward_log(vector<vector<double> > &transition, vector<vector<double> > &emission,  vector<double> &pi, vector<int> &observations, int stop, int num_threads);

/*
    The HMM backward algorithm uses dynamic programming to estimate the probablity of being in state i at time t, and
    seeing a particular sequence of observations after transitioning to the next state at time t + 1

    params:
        vector<vector<double> > &transition: a matrix representing the transition probabilities of the HMM
        vector<vector<double> > &emission: a matrix representing the probability of making an observation from a given state
        vector<double> &pi: a vector representing the probability of starting in a given state
        vector<int> &observations: a vector representing a sequence of observations
        int num_threads: number of threads to run in parallel
    return:
        vector<vector<double> >: a backward probability matrix that represent the probability of being in state i at time t
        and then seeing observations t+1,t+2,...,T, where T is the total number of observations (one observation at each time t)

*/
vector<vector<double> > backward_log(vector<vector<double> > &transition, vector<vector<double> > &emission, vector<double> &pi, vector<int> &observations, int num_threads);


/*
    Using the forward-backward algorithm, predict HMM parameters that best estimate a training set of 
    observations.

    params:
        HmmParams &params: starting parameters of an HMM to be trained
        vector<vector<int> > &training: a set of observations sequences with which to train the HMM
        int iterations: the number of iterations of training to go through
        int num_threads: number of threads to run in parallel

    return:
        HmmParams: a new set of HMM parameters that represent an improved estimate given the training data

*/
HmmParams baum_welch(HmmParams &params, vector<vector<int> > &training, int iterations, int num_threads);



#endif
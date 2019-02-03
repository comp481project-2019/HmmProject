#include <iostream>      
#include <vector>           
#include <stdlib.h>         
#include <time.h>          
#include <chrono>           
#include <fstream>          

using namespace std;
using namespace std::chrono;


struct HmmParams {
    vector<vector<double> > transition;
    vector<vector<double> > emission;
    vector<double> initial;
};

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
vector<vector<double> > forward(vector<vector<double> > &transition, vector<vector<double> > &emission,  vector<double> &pi, vector<int> &observations, int stop)
{
    int numCols = stop;
    int numRows = transition.size();
    int row, col, p;
    double stateProb, result;

    vector<vector<double> > resultMatrix(numRows, vector<double>(numCols)); // Defaults to zero initial value

    // initialization
    for (row = 0; row < numRows; row++){
        resultMatrix[row][0] = pi[row]*emission[row][observations[0]];
    }

    // calculate forward probabilities
    for (col = 1; col < numCols; col++)
    {
        for (row = 0; row < numRows; row++)
        {
            stateProb = 0;
            for (p = 0; p < numRows; p++)
            {
                stateProb = transition[p][row]*resultMatrix[p][col-1]*emission[row][observations[col]];
                resultMatrix[row][col] += stateProb;
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
vector<vector<double> > backward(vector<vector<double> > &transition, vector<vector<double> > &emission, vector<double> &pi, vector<int> &observations)
{
    int numCols = observations.size();
    int numRows = transition.size();
    int row, col, p;
    double stateProb, result;

    vector<vector<double> > resultMatrix(numRows, vector<double>(numCols)); // Defaults to zero initial value
    
    // initial probabilities
    for (row = 0; row < numRows; row++)
    {
        resultMatrix[row][numCols-1] = 1;
    }
    
    // calculate backward probabilities 
    for (col = numCols-2; col >= 0; col--)
    {
        for (row = 0; row < numRows; row++)
        {
            for (p = 0; p < numRows; p++)
            {
                resultMatrix[row][col] += transition[row][p]*emission[p][observations[col+1]]*resultMatrix[p][col+1];
            }
            
        }
    }
    
    /* 
        probabilities of returning to inital state. the sum should equal result of forward algorithm.
        This value is calculated for testing purposes only and is not yet implemented in the code.
    */
    float sum = 0;
    for (row = 0; row < numRows; row++)
    {
        sum += pi[row]*resultMatrix[row][0]*emission[row][observations[0]];
    } 

    //cout << sum << "\n";
    
    

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
HmmParams baum_welch(HmmParams &params, vector<vector<int> > &training, int iterations)
{
    
    int it;
    vector<vector<double> > transition(params.transition);
    vector<vector<double> > emission(params.emission);
    vector<double> initial(params.initial);
    int numStates = initial.size();
    int numObservs;

    // this represent the probability of an observation sequence to be trained on.
    float za;


    for (it = 0; it < iterations; it++)
    {
        // initialize vars
        vector<vector<double> > transitionUpdate(transition.size(), vector<double>(transition[0].size()));
        vector<vector<double> > emissionUpdate(emission.size(), vector<double>(emission[0].size()));
        vector<double> initialUpdate(initial.size());

        for (int o = 0; o < training.size(); o++){
            vector<int> observations = training[o];
            numObservs = observations.size();

            // get alpha
            vector<vector<double> > alpha = forward(transition, emission, initial, observations, observations.size());
            // given alpha, calculate the probability of seeing the observation sequence
            za = 0;
            for (int i = 0; i < alpha.size(); i++)
            {
                za += alpha[i][alpha[0].size()-1];
            }

            // get beta
            vector<vector<double> > beta = backward(transition, emission, initial, observations);

            // update initial
            for (int row = 0; row < numStates; row++)
            {
                initialUpdate[row] += alpha[row][0]*beta[row][0]/za;
            }

            // update emission probs
            for (int col = 0; col < numObservs; col++)
            {
                for (int row = 0; row < numStates; row++)
                {
                    emissionUpdate[row][observations[col]] += alpha[row][col]*beta[row][col]/za; 
                }
            
            }

            // update transition probs
            for (int i = 1; i < numObservs; i++)
            {
                for (int s1 = 0; s1 < numStates; s1++)
                {
                    for (int s2 = 0; s2 < numStates; s2++)
                    {
                        transitionUpdate[s1][s2] += alpha[s1][i-1]*transition[s1][s2]*emission[s2][observations[i]]*beta[s2][i]/za;
                    }
                }
            }

        }

        // create denominators for normalization
        double initialSum = 0;
        vector<double> transitionSum(numStates);
        vector<double> emissionSum(emission[0].size());
        // sum of initals
        for (int s = 0; s < numStates; s++)
        {
            initialSum += initialUpdate[s];
        }
        // sum of values in each tranistion row
        for (int row = 0; row < numStates; row++)
        {
            for (int col = 0; col < numStates; col++)
            {
                transitionSum[row] += transitionUpdate[row][col];
            }
        }
        // sum of values in each emssion row
        for (int row = 0; row < numStates; row++)
        {
            for (int col = 0; col < emissionSum.size(); col++)
            {
                emissionSum[row] += emissionUpdate[row][col];
            }
        }

        // normalization step:

        // 1. normailze initial
        for (int s = 0; s < numStates; s++)
        {
            initial[s] = initialUpdate[s]/initialSum;
        }

        // 2. normalize transition
        for (int row = 0; row < numStates; row++)
        {
            for (int col = 0; col < numStates; col++)
            {
                transition[row][col] = transitionUpdate[row][col]/transitionSum[row];
            }
        }

        // 3. normalize emission
        for (int row = 0; row < numStates; row++)
        {
            for (int col = 0; col < emissionSum.size(); col++)
            {
                emission[row][col] = emissionUpdate[row][col]/emissionSum[row];
            }
        }
        

    }

    HmmParams newParams;

    newParams.initial = initial;
    newParams.transition = transition;
    newParams.emission = emission;

    return newParams;

}


/*
    

    params:
        None

    return:
        int: 0 if program successfully executes
*/
int main()
{

    cout << "\nSerial HMM Implementation Incuding Forward, Backward, and Baum-Welch Algorithms:\n\n";
    // create simple testing model for HMM including inital (pi), transition, and emission probabilities 
    vector<double> pi {0.5, 0.5};
    vector<vector<double> > transition {{ 0.5, 0.5 },
                                    {0.3, 0.7}};

    vector<vector<double> > emission { { 0.3, 0.5, 0.2 },
                                    { 0.2, 0.2, 0.6 }};

    
    // define a sequence of observations
    vector<int> observations {0,1,1};

    cout << "HMM Initial Probabilites:\n";
    print_double_vector(pi);
    cout << "HMM Transition Probabilites:\n";
    print_double_matrix(transition);
    cout << "HMM Emission Probabilites:\n";
    print_double_matrix(emission);
    cout << "Observation Sequence:\n";
    print_int_vector(observations);

    // calclate the forward probability matrix
    vector<vector<double> > forwardResult = forward(transition, emission, pi, observations, observations.size());
    cout << "\nForward Probabilities:\n";
    print_double_matrix(forwardResult);

    // calculate the backward probability matrix
    vector<vector<double> > backwardResult = backward(transition, emission, pi, observations);
    cout << "Backward Probailities:\n";
    print_double_matrix(backwardResult);


    // define a training set of observations
    vector<vector<int> > training{{0,2,1,0,2,0,1,0,0,2,1,2,0,0,1,1,1,2,2,0,2,2,0,1},
                        {2,0,2,1,2,1,0,0,0,1,2},
                        {1,0,2,0,2}};

    /* 
        using the initial, transition, and emission probabilities from before, initialize an HMM
        parameter structure
    */
    HmmParams params;
    params.transition = transition;
    params.emission = emission;
    params.initial = pi;

    // perform the baum_welch algorithm with the HMM params, the training data, and using 10 iterations 
    HmmParams newParams = baum_welch(params, training, 10);

    // update the HMM params with the new parameters from training
    params = newParams;

    cout << "Trained HMM Initial Probabilites:\n";
    print_double_vector(params.initial);
    cout << "Trained HMM Transition Probabilites:\n";
    print_double_matrix(params.transition);
    cout << "Trained HMM Emission Probabilites:\n";
    print_double_matrix(params.emission);



	return 0;
}

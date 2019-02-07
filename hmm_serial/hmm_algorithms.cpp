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


struct HmmData {
    int numHmmInputs = 0;
    vector<HmmParams> hmmParamList;
    vector<vector<vector<int> > > trainingSets;
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
            random = 1 + static_cast<float>(rand()) /(static_cast<float>(RAND_MAX/(10 - 1))); // ensures the random number is in the input range
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

HmmParams generate_random_hmm(int numStates, int numEmissions)
{
    HmmParams params;
    params.initial = generate_probability_vector(numStates);
    params.transition = generate_probability_matrix(numStates, numStates);
    params.emission = generate_probability_matrix(numStates, numEmissions);


    return params;
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
        vector<double> emissionSum(emission.size());
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

        //sum of values in each emssion row
        for (int row = 0; row < numStates; row++)
        {
            for (int col = 0; col < emissionUpdate[0].size(); col++)
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
            for (int col = 0; col < emissionUpdate[0].size(); col++)
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
    
    Runs the baum welch algorithm and determines whether or not to print the results

    params:
        None

    return:
        int: 0 if program successfully executes
*/
int main()
{
    srand (time(NULL)); // set seed for random generator 
    bool print = true; // print models to standard output?
    int readFile;
    HmmData fData;

    
    cout << "Input 1 to read from file. Otherwise, generate random parameters:\n";
    cin >> readFile;

    if (readFile == 1)
    {
        fData = read_file_data();  // read in data from file
    }
    else 
    {
        vector<int> sizes({10, 20, 15, 5}); // these are the sizes for random observation vectors
        fData.hmmParamList.push_back(generate_random_hmm(10, 7)); //add a randomized hmm to the list
        fData.trainingSets.push_back(generate_training_set(7, sizes)); // add a randomized training set to the list
        fData.numHmmInputs++;   // indicate that a new hmm input has been added;
    }

    HmmParams params;
    vector<vector<int> > training;
    for (int i = 0; i < fData.numHmmInputs; i++)
    {
        params = fData.hmmParamList[i];
        training = fData.trainingSets[i];

        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        HmmParams newParams = baum_welch(params, training, 10); // run baum-welch with 10 iterations
        high_resolution_clock::time_point t2 = high_resolution_clock::now();

        if (print)
        {
            cout << "\nHMM TRAINING MODEL " << i << "\n\n";
            cout << "HMM Initial Probabilites:\n";
            print_double_vector(params.initial);
            cout << "HMM Transition Probabilites:\n";
            print_double_matrix(params.transition);
            cout << "HMM Emission Probabilites:\n";
            print_double_matrix(params.emission);
            cout << "Observation Training Data:\n";
            print_int_matrix(training);

            

            cout << "\nHMM Trained Initial Probabilites:\n";
            print_double_vector(newParams.initial);
            cout << "HMM Trained Transition Probabilites:\n";
            print_double_matrix(newParams.transition);
            cout << "HMM Trained Emission Probabilites:\n";
            print_double_matrix(newParams.emission);
            cout << "\n\n";

        }
        cout << "Serial Exectution Time: " << duration_cast<microseconds>( t2 - t1 ).count() << " microseconds" << endl;

    }

	return 0;
}

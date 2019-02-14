#include <iostream>      
#include <vector>           
#include <stdlib.h>         
#include <time.h>          
#include <omp.h>           
#include <fstream> 
#include <cmath>  
#include "hmm_algorithms_para.h"

int NUM_ITERATIONS = 1;
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
    bool print = false; // print models to standard output?
    int readFile;
    double t1, t2;
    HmmData fData;


    cout << "Input 1 to read from file. Otherwise, generate random parameters:\n";
    cin >> readFile;

    if (readFile == 1)
    {
        print = true;
        fData = read_file_data();  // read in data from file
    }
    else 
    {
        vector<int> sizes1({10}); // these are the sizes for random observation vectors
        fData.hmmParamList.push_back(generate_random_hmm(1024, 3)); //add a randomized hmm to the list
        fData.trainingSets.push_back(generate_training_set(3, sizes1)); // add a randomized training set to the list
        fData.numHmmInputs++;   // indicate that a new hmm input has been added;

    }

    HmmParams params;
    vector<vector<int> > training;
    for (int i = 0; i < fData.numHmmInputs; i++)
    {
        t2 = 0;
        for (int j = 0; j < NUM_ITERATIONS; j++)
        {
            
            params = fData.hmmParamList[i];
            training = fData.trainingSets[i];

            t1 = omp_get_wtime();
            HmmParams newParams = baum_welch(params, training, 10); // run baum-welch with 10 iterations
            t2 += omp_get_wtime()-t1;

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
        }
        cout << "Parallel Exectution Time: " << ( t2/NUM_ITERATIONS) << " seconds" << endl;

    }

	return 0;
}
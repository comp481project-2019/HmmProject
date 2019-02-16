#include <iostream>      
#include <vector>           
#include <stdlib.h>         
#include <time.h>          
#include <omp.h>           
#include <fstream> 
#include <cmath>  
#include "hmm_algorithms_para.h"
#include "hmm_algorithms.h"

int NUM_ITERATIONS = 5;
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
    double t1, para_baum_welch_duration, para_forward_duration, para_backward_duration;
    double serial_baum_welch_duration, serial_forward_duration, serial_backward_duration;
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
        vector<int> sizes1({1000}); // these are the sizes for random observation vectors
        fData.hmmParamList.push_back(generate_random_hmm(1024, 32)); //add a randomized hmm to the list
        fData.trainingSets.push_back(generate_training_set(32, sizes1)); // add a randomized training set to the list
        fData.numHmmInputs++;   // indicate that a new hmm input has been added;

    }

    HmmParams params;
    vector<vector<int> > training;
    for (int i = 0; i < fData.numHmmInputs; i++)
    {
        para_baum_welch_duration = 0;
        para_forward_duration = 0;
        para_backward_duration = 0;
        serial_baum_welch_duration = 0;
        serial_forward_duration = 0;
        serial_backward_duration = 0;
        for (int j = 0; j < NUM_ITERATIONS; j++)
        {
            
            params = fData.hmmParamList[i];
            training = fData.trainingSets[i];

            convert_matrix_to_log_space_para(params.transition);
            convert_matrix_to_log_space_para(params.emission);
            convert_vector_to_log_space_para(params.initial);
            
            t1 = omp_get_wtime();
            forward_log(params.transition, params.emission, params.initial, training[0],  training[0].size()); // run baum-welch with 10 iterations
            para_forward_duration += omp_get_wtime()-t1;

            t1 = omp_get_wtime();
            forward_serial_log(params.transition, params.emission, params.initial, training[0],  training[0].size()); // run baum-welch with 10 iterations
            serial_forward_duration += omp_get_wtime()-t1;

            t1 = omp_get_wtime();
            backward_log(params.transition, params.emission, params.initial, training[0]); // run baum-welch with 10 iterations
            para_backward_duration += omp_get_wtime()-t1;

            t1 = omp_get_wtime();
            backward_serial_log(params.transition, params.emission, params.initial, training[0]); // run baum-welch with 10 iterations
            serial_backward_duration += omp_get_wtime()-t1;

            convert_matrix_from_log_space_para(params.transition);
            convert_matrix_from_log_space_para(params.emission);
            convert_vector_from_log_space_para(params.initial);

            t1 = omp_get_wtime();
            HmmParams newParamsSerial = baum_welch_serial(params, training, 5); // run baum-welch with 10 iterations
            serial_baum_welch_duration += omp_get_wtime()-t1;


            t1 = omp_get_wtime();
            HmmParams newParams = baum_welch(params, training, 5); // run baum-welch with 10 iterations
            para_baum_welch_duration += omp_get_wtime()-t1;

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

                cout << "\nSerial HMM Trained Initial Probabilites:\n";
                print_double_vector(newParamsSerial.initial);
                cout << "HMM Trained Transition Probabilites:\n";
                print_double_matrix(newParamsSerial.transition);
                cout << "HMM Trained Emission Probabilites:\n";
                print_double_matrix(newParamsSerial.emission);
                cout << "\n\n";

                cout << "\nParallel HMM Trained Initial Probabilites:\n";
                print_double_vector(newParams.initial);
                cout << "HMM Trained Transition Probabilites:\n";
                print_double_matrix(newParams.transition);
                cout << "HMM Trained Emission Probabilites:\n";
                print_double_matrix(newParams.emission);
                cout << "\n\n";

            }
        }
        serial_baum_welch_duration = serial_baum_welch_duration/NUM_ITERATIONS;
        para_baum_welch_duration = para_baum_welch_duration/NUM_ITERATIONS;
        double baum_welch_speedup = serial_baum_welch_duration/para_baum_welch_duration;
        cout << "Baum-Welch:\n";
        cout << "\tSerial Exectution Time: " << (serial_baum_welch_duration) << " seconds" << endl;
        cout << "\tParallel Exectution Time: " << (para_baum_welch_duration) << " seconds" << endl;
        cout << "\tSpeedup: " << (baum_welch_speedup) << endl << endl;

        serial_forward_duration = serial_forward_duration/NUM_ITERATIONS;
        para_forward_duration = para_forward_duration/NUM_ITERATIONS;
        double forward_speedup = serial_forward_duration/para_forward_duration;
        cout << "Forward:\n";
        cout << "\tSerial Exectution Time: " << (serial_forward_duration) << " seconds" << endl;
        cout << "\tParallel Exectution Time: " << (para_forward_duration) << " seconds" << endl;
        cout << "\tSpeedup: " << (forward_speedup) << endl << endl;

        serial_backward_duration = serial_backward_duration/NUM_ITERATIONS;
        para_backward_duration = para_backward_duration/NUM_ITERATIONS;
        double backward_speedup = serial_backward_duration/para_backward_duration;
        cout << "Backward:\n";
        cout << "\tSerial Exectution Time: " << (serial_backward_duration) << " seconds" << endl;
        cout << "\tParallel Exectution Time: " << (para_backward_duration) << " seconds" << endl;
        cout << "\tSpeedup: " << (backward_speedup) << endl;


    }

	return 0;
}
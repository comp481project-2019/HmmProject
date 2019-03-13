#include <iostream>      
#include <vector>           
#include <stdlib.h>         
#include <time.h>          
#include <omp.h>           
#include <fstream> 
#include <cmath>  
#include "hmm_algorithms_para.h"
#include "hmm_algorithms.h"


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
    int readFile, default_vals, num_states, num_emissions, num_observations, NUM_ITERATIONS;
    double t1, para_baum_welch_duration, para_forward_duration, para_backward_duration;
    double serial_baum_welch_duration, serial_forward_duration, serial_backward_duration;
    HmmData fData;
    vector<int> threads;

    NUM_ITERATIONS = 1;

    ofstream outputfile;  
   
    cout << "Input 1 to read from file. Otherwise, generate random parameters:\n";
    cin >> readFile;

    if (readFile == 1)
    {
        print = true;
        fData = read_file_data();  // read in data from file
    }
    else 
    {  
        

        cout << "Input 1 to use default values:\n";
        cin >> default_vals;

        if (default_vals == 1)
        {
            NUM_ITERATIONS = 5;
            threads = vector<int>({2, 4, 8, 16, 32, 52, 64, 128});
            num_states = 1024;
            num_emissions = 32;
            num_observations = 1000;
        } else {

            cout << "Enter the number of iterations over which to average timing:\n";
            cin >> NUM_ITERATIONS;

            int thread_counts;
            cout << "Enter the number of states in the HMM:\n";
            cin >> num_states;

            cout << "Enter the number of emissions in the HMM:\n";
            cin >> num_emissions;

            cout << "Enter the number of observations for the HMM:\n";
            cin >> num_observations;

            cout << "Enter the number of thread counts to test:\n";
            cin >> thread_counts;

            for (int t = 0; t < thread_counts; t++)
            {
                int val;
                cout << "Enter thread count " << t << ":\n";
                cin >> val;

                threads.push_back(val);
            }

            
        }

        outputfile.open ("hmm_timing.csv", ios::out | ios::app);

        vector<int> sizes1({num_observations}); // these are the sizes for random observation vectors
        fData.hmmParamList.push_back(generate_random_hmm(num_states, num_emissions)); //add a randomized hmm to the list
        fData.trainingSets.push_back(generate_training_set(num_emissions, sizes1)); // add a randomized training set to the list
        fData.numHmmInputs++;   // indicate that a new hmm input has been added;

    }

    HmmParams params;
    vector<vector<int> > training;

    for (int i = 0; i < fData.numHmmInputs; i++)
    {
        serial_baum_welch_duration = 0;
        serial_forward_duration = 0;
        serial_backward_duration = 0;
        
        vector<double> baum_welch_durations(threads.size());
        vector<double> forward_durations(threads.size());
        vector<double> backward_durations(threads.size());
        
        for (int j = 0; j < NUM_ITERATIONS; j++)
        {
            
            params = fData.hmmParamList[i];
            training = fData.trainingSets[i];

            convert_matrix_to_log_space_para(params.transition, 4);
            convert_matrix_to_log_space_para(params.emission, 4);
            convert_vector_to_log_space_para(params.initial, 4);

            t1 = omp_get_wtime();
            forward_serial_log(params.transition, params.emission, params.initial, training[0],  training[0].size()); // run baum-welch with 10 iterations
            serial_forward_duration += omp_get_wtime()-t1;

            t1 = omp_get_wtime();
            backward_serial_log(params.transition, params.emission, params.initial, training[0]); // run baum-welch with 10 iterations
            serial_backward_duration += omp_get_wtime()-t1;


            for (int t = 0; t < threads.size(); t++)
            {
                t1 = omp_get_wtime();
                forward_log(params.transition, params.emission, params.initial, training[0],  training[0].size(), threads[t]); // run baum-welch with 10 iterations
                forward_durations[t] += omp_get_wtime()-t1;

                
                t1 = omp_get_wtime();
                backward_log(params.transition, params.emission, params.initial, training[0], threads[t]); // run baum-welch with 10 iterations
                backward_durations[t] += omp_get_wtime()-t1;
            }

            
            convert_matrix_from_log_space_para(params.transition, 4);
            convert_matrix_from_log_space_para(params.emission, 4);
            convert_vector_from_log_space_para(params.initial, 4);



            t1 = omp_get_wtime();
            HmmParams newParamsSerial = baum_welch_serial(params, training, 5); // run baum-welch with 10 iterations
            serial_baum_welch_duration += omp_get_wtime()-t1;

            HmmParams newParams;

            for (int t = 0; t < threads.size(); t++)
            {
                t1 = omp_get_wtime();
                newParams = baum_welch(params, training, 5, threads[t]); // run baum-welch with 10 iterations
                baum_welch_durations[t] += omp_get_wtime()-t1;
            }

           

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
        serial_forward_duration = serial_forward_duration/NUM_ITERATIONS;
        serial_backward_duration = serial_backward_duration/NUM_ITERATIONS;

        if (readFile != 1)
        {
            outputfile << 1 << "," << num_states << "," << num_emissions << "," << num_observations << ",";
            outputfile << "baum-welch," << serial_baum_welch_duration << "," << serial_baum_welch_duration << ",";
            outputfile << 1 << "," << 1 << "\n";

            outputfile << 1 << "," << num_states << "," << num_emissions << "," << num_observations << ",";
            outputfile << "forward," << serial_forward_duration << "," << serial_forward_duration << ",";
            outputfile << 1 << "," << 1 << "\n";

            outputfile << 1 << "," << num_states << "," << num_emissions << "," << num_observations << ",";
            outputfile << "backward," << serial_backward_duration << "," << serial_backward_duration << ",";
            outputfile << 1 << "," << 1 << "\n";
        }
        

        for (int t = 0; t < threads.size(); t++) 
        {
            double para_baum_welch_duration = baum_welch_durations[t]/NUM_ITERATIONS;
            double baum_welch_speedup = serial_baum_welch_duration/para_baum_welch_duration;
            double baum_welch_efficiency = baum_welch_speedup/threads[t];

            cout << "\n\nTHREAD COUNT: " << threads[t] << endl;
            cout << "Baum-Welch:\n";
            cout << "\tSerial Exectution Time: " << (serial_baum_welch_duration) << " seconds" << endl;
            cout << "\tParallel Exectution Time: " << (para_baum_welch_duration) << " seconds" << endl;
            cout << "\tSpeedup: " << (baum_welch_speedup) << endl;
            cout << "\tEfficiency: " << (baum_welch_efficiency) << endl << endl;

            
            double para_forward_duration = forward_durations[t]/NUM_ITERATIONS;
            double forward_speedup = serial_forward_duration/para_forward_duration;
            double forward_efficiency = forward_speedup/threads[t];
            cout << "Forward:\n";
            cout << "\tSerial Exectution Time: " << (serial_forward_duration) << " seconds" << endl;
            cout << "\tParallel Exectution Time: " << (para_forward_duration) << " seconds" << endl;
            cout << "\tSpeedup: " << (forward_speedup) << endl;
            cout << "\tSpeedup: " << (forward_efficiency) << endl << endl;


            
            double para_backward_duration = backward_durations[t]/NUM_ITERATIONS;
            double backward_speedup = serial_backward_duration/para_backward_duration;
            double backward_efficiency = backward_speedup/threads[t];
            cout << "Backward:\n";
            cout << "\tSerial Exectution Time: " << (serial_backward_duration) << " seconds" << endl;
            cout << "\tParallel Exectution Time: " << (para_backward_duration) << " seconds" << endl;
            cout << "\tSpeedup: " << (backward_speedup) << endl;
            cout << "\tEfficiency: " << (backward_efficiency) << endl;

            if (readFile != 1)
            {
                outputfile << threads[t] << "," << num_states << "," << num_emissions << "," << num_observations << ",";
                outputfile << "baum-welch," << serial_baum_welch_duration << "," << para_baum_welch_duration << ",";
                outputfile << baum_welch_speedup << "," << baum_welch_efficiency << "\n";

                outputfile << threads[t] << "," << num_states << "," << num_emissions << "," << num_observations << ",";
                outputfile << "forward," << serial_forward_duration << "," << para_forward_duration << ",";
                outputfile << forward_speedup << "," << forward_efficiency << "\n";

                outputfile << threads[t] << "," << num_states << "," << num_emissions << "," << num_observations << ",";
                outputfile << "backward," << serial_backward_duration << "," << para_backward_duration << ",";
                outputfile << backward_speedup << "," << backward_efficiency << "\n";
            }


             
        }


    }

    if (readFile != 1)
    {
        outputfile.close();
    }
    

	return 0;
}
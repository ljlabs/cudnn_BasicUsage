#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <string>
#include <sstream>
#include <curand.h>

using namespace std;
__device__ __managed__ float weightedinputs[25]; // used as list of input neuron
__device__ __managed__ float weights[25]; // used as list of neuron conection weigths
__device__ __managed__ int inputs[25]; // used as list of neuron conection weigths
__device__ __managed__ float output = 0; // used to return output
__device__ __managed__ int expctd = 0; // used to return output

__global__ void mulWeightsAndInputs() {

	weightedinputs[threadIdx.x] = weights[threadIdx.x] * inputs[threadIdx.x];

}

__global__ void feedbackward(){			// trains the weights
	float lr = 0.3;
	float error = (expctd - output);
	weights[threadIdx.x] = weights[threadIdx.x] + error * inputs[threadIdx.x] * lr;
}

void init_Weights(){
	for (int i = 0; i < 25; i++){
		weights[i] = rand() % 10 + 1;
	}
}

void feed_forward(){
	double tot = 0;
	for (int i = 0; i < 25; i++){
		tot = tot + weightedinputs[i];	
		//cout << tot << endl;
	}
	long sqtot = pow(tot, 2);
	output = 0.5*((tot / (1 + sqrt(sqtot))) + 1);

}
void cp_To_Dev(int data[25]){
	for (int i = 0; i < 25; i++){
		inputs[i] = data[i];
	}
}


int main()
{
	/*generate some training data*/
	int expected[5] = { 1, 0, 0, 0, 0 };	// lets learn the letter a
	int trainingData[5][25] = {
		{
			0, 0, 1, 0, 0,
			0, 1, 0, 1, 0,
			1, 0, 0, 0, 1,
			1, 1, 1, 1, 1,
			1, 0, 0, 0, 1,
		},
		{
			1, 1, 1, 1, 0,
			1, 0, 0, 0, 1,
			1, 1, 1, 1, 0,
			1, 0, 0, 0, 1,
			1, 1, 1, 1, 0
		},
		{
			0, 1, 1, 1, 0,
			1, 0, 0, 0, 1,
			1, 0, 0, 0, 0,
			1, 0, 0, 0, 1,
			0, 1, 1, 1, 0
		}, 
		{
			1, 1, 1, 1, 0,
			1, 0, 0, 0, 1,
			1, 0, 0, 0, 1,
			1, 0, 0, 0, 1,
			1, 1, 1, 1, 0
		},
		{
			1, 1, 1, 1, 1,
			1, 0, 0, 0, 0,
			1, 1, 1, 1, 0,
			1, 0, 0, 0, 0,
			1, 1, 1, 1, 1
		}
	};
	// initilise the weights
	init_Weights();

	for (int epoc = 0; epoc < 1000; epoc++){	// i want this to run 1000 time to ensure a good train
		for (int nRow = 0; nRow < 5; nRow++){

			// copy inputs to device
			cp_To_Dev(trainingData[nRow]);
			mulWeightsAndInputs << < 1, 25 >> >();
			cudaDeviceSynchronize();
			feed_forward();
			// learn
			expctd = expected[nRow];
			feedbackward << < 1, 25 >> >();
			cudaDeviceSynchronize();
		}
	}

	int TestData[4][25] = {
		{// broken A
			0, 0, 0, 0, 0,
			0, 1, 0, 1, 0,
			1, 0, 0, 0, 1,
			1, 1, 1, 1, 1,
			0, 1, 0, 0, 1,
		}, { // a B also broken
			1, 1, 1, 1, 0,
			1, 0, 1, 0, 1,
			1, 1, 0, 0, 1,
			1, 0, 0, 0, 1,
			1, 1, 1, 1, 0
		}, { //  a very broken A
			0, 0, 1, 0, 0,
			0, 1, 0, 1, 0,
			1, 0, 1, 0, 1,
			1, 1, 0, 0, 1,
			1, 0, 0, 0, 1,
		},
	};
	
	for (int i = 0; i < 3; i++){
		// test the network
		cp_To_Dev(TestData[i]);
		mulWeightsAndInputs << < 1, 25 >> >();
		cudaDeviceSynchronize();
		feed_forward();
		cout << output << endl;
	}

	// this just forces the program to wait until i have completed looking at the ouput
	int wait;
	cin >> wait;
    return 0;
}

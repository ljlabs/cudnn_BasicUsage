#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <C:\Users\jorda_000\Documents\cuda\libs\cudnn-6.5-win-v2\cudnn.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <sstream>

using namespace std;
__device__ __managed__ int ret[1000]; // managed varaible

__global__ void AplusB(int a, int b) {
	ret[threadIdx.x] = threadIdx.x * a+ b + threadIdx.x; 
}


int main()
{
	AplusB <<< 1, 10 >>>(5, 10);
	cudaDeviceSynchronize();
	for (int i = 0; i < 1000; i++){
		cout << ret[i];
	}
	int a;
	cin >> a;
    return 0;
}

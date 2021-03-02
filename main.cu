#include "greeter.h"

#include <stdio.h>

/*
 * A simple introduction to programming in CUDA. This program prints "Hello
 * World from GPU! from 10 CUDA threads running on the GPU.
 */

__global__ void helloFromGPU() {
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Hello World from GPU thread %d!\n", idx);
}

int main(int argc, char **argv) {
	greet("Pinco");

	helloFromGPU<<<1, 10>>>();
	cudaDeviceSynchronize(); // try commenting this line
	return 0;
}


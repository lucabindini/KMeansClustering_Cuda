#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda.h>
#include <iostream>
#include <random>
#include <chrono>
#include <cmath>

#define N 1024
#define K 2
#define DIM 2

static void CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux(const char *file, unsigned line,
		const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
			<< err << ") at " << file << ":" << line << std::endl;
	exit(1);
}

void generatePoints(float * points) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < DIM; j++) {
			points[j*N+i] = rand() / (float) RAND_MAX;
		}
	}
}

void generateCentroids(float * centroids, float * points) {
	for (int i = 0; i < K; i++) {
		for (int j = 0; j < DIM; j++) {
			centroids[j*K+i] = points[j*N+i];
		}
	}
}

void printPoints(float * points, int num) {
	for (int i = 0; i < num; i++) {
		for (int j = 0; j < DIM; j++)
			printf("%f ", points[j*num + i]);
		printf("\n");
	}
}

void printIds(int * ids) {
	for (int i = 0; i < N; i++)
		printf("%d: %d\n", i, ids[i]);
}

__device__ float pointDistance(float * points, float * centroids, int p, int c) {
	float sum = 0;
	for (int j = 0; j < DIM; j++)
		sum += pow((points[j * N + p] - centroids[j * K + c]), 2);
	return sqrt(sum);
}

__global__ void assignCentroid(float * points, float * centroids, int * ids) {
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	if (p < N) {
		int nearestId = 0;
		float minDistance = pointDistance(points, centroids, p, 0);
		for (int c = 1; c < K; c++) {
			float distance = pointDistance(points, centroids, p, c);
			if (distance < minDistance) {
				minDistance = distance;
				nearestId = c;
			}
		}
		ids[p] = nearestId;
	}
}

int main(int argc, char **argv) {
	/*
	auto start = std::chrono::system_clock::now();
	KMeans kmeans(5, 10000, 20, 10);
	kmeans.execute();
	auto end = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
	std::cout<<"Tempo: "<< elapsed.count();
	*/
	srand(time(NULL));
	float * points, * devPoints; // points[i][j] = points[i*J+j] I=n J=dim
	float * centroids, * devCentroids;
	int * ids, * devIds;

	// Generate points
	points = (float *) malloc(N * DIM * sizeof(float));
	generatePoints(points);
	// Initialize centroids
	centroids = (float *) malloc(K * DIM * sizeof(float));
	generateCentroids(centroids, points);
	// Inizialize id vectors
	ids = (int *) malloc(N * sizeof(int));

	//printPoint(points, n, dim);

	// Copy all data to device
	CUDA_CHECK_RETURN(cudaMalloc((void **) &devPoints, N * DIM * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMemcpy((void **) devPoints, (void **) points, N * DIM * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMalloc((void **) &devCentroids, K * DIM * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMemcpy((void **) devCentroids, (void **) centroids, K * DIM * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMalloc((void **) &devIds, N * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMemcpy((void **) devIds, (void **) ids, N * sizeof(int), cudaMemcpyHostToDevice));

	assignCentroid<<<1, N>>>(devPoints, devCentroids, devIds);

	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMemcpy((void **) points, (void **) devPoints, N * DIM * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy((void **) centroids, (void **) devCentroids, K * DIM * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy((void **) ids, (void **) devIds, N * sizeof(int), cudaMemcpyDeviceToHost));

	printPoints(points, N);
	printIds(ids);
	puts("Done");
}

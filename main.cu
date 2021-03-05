#include <cuda.h>
#include <iostream>
#include <random>
#include <chrono>

#define N 4096
#define K 256
#define DIM 2
#define MAX_ITER 256

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

void generatePoints(float * points, std::mt19937 gen) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < DIM; j++) {
			points[j*N+i] = std::generate_canonical<float, 10>(gen);
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

__global__ void assignCentroid(float * points, float * centroids, int * ids, bool * hasChanged) {
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
		if (ids[p] != nearestId) {
			ids[p] = nearestId;
			*hasChanged = true;
		}
	}
}

__global__ void moveCentroid(float * points, float * centroids, int * ids) {
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	if (c < K) {
		for (int d = 0; d < DIM; d++)
			centroids[K * d + c] = 0;
		int numPoints = 0;
		for (int p = 0; p < N; p++) {
			if (ids[p] == c) {
				//printf("point %d assigned to centroid %d\n", p, c);
				for (int d = 0; d < DIM; d++)
					centroids[K * d + c] += points[N * d + p];
				numPoints++;
			}
		}
		if (numPoints != 0) {
			for (int d = 0; d < DIM; d++)
				centroids[K * d + c] /= numPoints;
		}
		/*
	    for (int d = 0; d < DIM; d++)
	    	printf("coord %d of centroid %d: %f; ", d, c, centroids[K * d + c]);
		 */
	}
}

void kMeans(float * points, float * centroids, int * ids) {
	bool * hasChanged, * devHasChanged;
	hasChanged = (bool *) malloc(sizeof (bool));
	*hasChanged = true;
	CUDA_CHECK_RETURN(cudaMalloc((void **) &devHasChanged, sizeof(bool)));
	for (int i = 0; i < MAX_ITER && *hasChanged; i++) {
		*hasChanged = false;
		CUDA_CHECK_RETURN(cudaMemcpy((void **) devHasChanged, (void **) hasChanged, sizeof(bool), cudaMemcpyHostToDevice));
		assignCentroid<<<N/512, 512>>>(points, centroids, ids, devHasChanged);
		cudaDeviceSynchronize();
		moveCentroid<<<1, K>>>(points, centroids, ids);
		cudaDeviceSynchronize();
		CUDA_CHECK_RETURN(cudaMemcpy((void **) hasChanged, (void **) devHasChanged, sizeof(bool), cudaMemcpyDeviceToHost));
		//printf("%d\n", *hasChanged);
	}
}

int main(int argc, char **argv) {
	std::random_device rd;
	std::mt19937 gen(rd());
	float * points, * devPoints; // points[i][j] = points[i*J+j] I=n J=dim
	float * centroids, * devCentroids;
	int * ids, * devIds;

	// Allocate host memory
	points = (float *) malloc(N * DIM * sizeof(float));
	centroids = (float *) malloc(K * DIM * sizeof(float));
	ids = (int *) malloc(N * sizeof(int));

	// Allocate device memory
	CUDA_CHECK_RETURN(cudaMalloc((void **) &devPoints, N * DIM * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void **) &devCentroids, K * DIM * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void **) &devIds, N * sizeof(float)));

	int time = 0;
	int numTests = 10;
	for (int i = 0; i < numTests; i++) {
		// Generate points and centroids on host
		generatePoints(points, gen);
		generateCentroids(centroids, points);

		// Copy data to device
		CUDA_CHECK_RETURN(cudaMemcpy((void **) devPoints, (void **) points, N * DIM * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy((void **) devCentroids, (void **) centroids, K * DIM * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy((void **) devIds, (void **) ids, N * sizeof(int), cudaMemcpyHostToDevice));

		// Start timer
		auto start = std::chrono::system_clock::now();

		kMeans(devPoints, devCentroids, devIds);

		// Stop timer
		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
		time += elapsed.count();

		// Copy data back to host
		/*CUDA_CHECK_RETURN(cudaMemcpy((void **) centroids, (void **) devCentroids, K * DIM * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_CHECK_RETURN(cudaMemcpy((void **) ids, (void **) devIds, N * sizeof(int), cudaMemcpyDeviceToHost));*/
	}
	std::cout << "Time: "<< time/numTests << std::endl;

	//printPoints(points, N);
	//printIds(ids);
	//printPoints(centroids, K);

}

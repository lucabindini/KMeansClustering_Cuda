#include <cuda.h>
#include <iostream>
#include <random>
#include <chrono>

#define N 4096
#define K 256
#define DIM 2
#define MAX_ITER 64

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

__device__ float pointDistance(float * points, float * centroids, int p, int c) {
	float sum = 0;
	for (int j = 0; j < DIM; j++)
		sum += pow((points[j * N + p] - centroids[j * K + c]), 2);
	return sqrt(sum);
}

__global__ void kMeansKernel(float * points, float * centroids, float * newCentroids, unsigned int * pointsPerCluster) {
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("\t\tThread n. %d\n", p);
	if (p < N) {
		if (p < K) {
			for (int d = 0; d < DIM; d++)
				newCentroids[p * K + d] = 0;
			pointsPerCluster[p] = 0;
		}
		int nearestId = 0;
		float minDistance = pointDistance(points, centroids, p, 0);
		float distance;
		for (int c = 1; c < K; c++) {
			distance = pointDistance(points, centroids, p, c);
			if (distance < minDistance) {
				minDistance = distance;
				nearestId = c;
			}
		}
		atomicInc(&(pointsPerCluster[nearestId]), N);
		for (int d = 0; d < DIM; d++) {
			//printf("\t\t\td = %d\n", d);
			atomicAdd(&(newCentroids[nearestId * K + d]), points[d * N + p]);
		}
		__threadfence(); // wait for pointsPerCluster
		if (p < K) {
			for (int d = 0; d < DIM; d++)
				centroids[p * K + d] = newCentroids[p * K + d] / pointsPerCluster[p];
			//printf("\t\t%u points in cluster %d\n", pointsPerCluster[p], p);
		}
	}
}

void kMeans(float * points, float * centroids) {
	float * devNewCentroids;
	unsigned int * devPointsPerCluster;
	CUDA_CHECK_RETURN(cudaMalloc((void **) &devNewCentroids, K * DIM * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void **) &devPointsPerCluster, K * sizeof (unsigned int)));
	for (int i = 0; i < MAX_ITER; i++) {
		//printf("\titeration n. %d\n", i);
		kMeansKernel<<<ceil(N/(float)512), 512>>>(points, centroids, devNewCentroids, devPointsPerCluster);
		cudaDeviceSynchronize();
	}
	CUDA_CHECK_RETURN(cudaFree(devNewCentroids));
	CUDA_CHECK_RETURN(cudaFree(devPointsPerCluster));
}

int main(int argc, char **argv) {
	std::random_device rd;
	std::mt19937 gen(rd());
	float * points, * devPoints; // points[i][j] = points[i*J+j] I=n J=dim
	float * centroids, * devCentroids;

	// Allocate host memory
	points = (float *) malloc(N * DIM * sizeof(float));
	centroids = (float *) malloc(K * DIM * sizeof(float));

	// Allocate device memory
	CUDA_CHECK_RETURN(cudaMalloc((void **) &devPoints, N * DIM * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void **) &devCentroids, K * DIM * sizeof(float)));

	int time = 0;
	int numTests = 16;
	for (int i = 0; i < numTests; i++) {
		printf("Test n. %d\n", i);

		// Generate points and centroids on host
		generatePoints(points, gen);
		generateCentroids(centroids, points);

		// Copy data to device
		CUDA_CHECK_RETURN(cudaMemcpy((void *) devPoints, (void *) points, N * DIM * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy((void *) devCentroids, (void *) centroids, K * DIM * sizeof(float), cudaMemcpyHostToDevice));

		// Start timer
		auto start = std::chrono::system_clock::now();

		kMeans(devPoints, devCentroids);

		// Stop timer
		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
		time += elapsed.count();

		// Copy data back to host
		//CUDA_CHECK_RETURN(cudaMemcpy((void **) centroids, (void **) devCentroids, K * DIM * sizeof(float), cudaMemcpyDeviceToHost));
	}
	std::cout << "Time: "<< time/numTests << std::endl;

	//printPoints(points, N);
	//printPoints(centroids, K);

	free(points);
	free(centroids);
	cudaFree(devPoints);
	cudaFree(devCentroids);
}

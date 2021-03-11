#include <cuda.h>
#include <iostream>
#include <random>
#include <chrono>

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

void generatePoints(float * points, std::mt19937 gen, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < DIM; j++) {
			points[j*n+i] = std::generate_canonical<float, 10>(gen);
		}
	}
}

void generateCentroids(float * centroids, float * points, int n, int k) {
	for (int i = 0; i < k; i++) {
		for (int j = 0; j < DIM; j++) {
			centroids[j*k+i] = points[j*n+i];
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

__device__ float pointDistance(float * points, float * centroids, int p, int c, int n, int k) {
	float sum = 0;
	for (int j = 0; j < DIM; j++)
		sum += pow((points[j * n + p] - centroids[j * k + c]), 2);
	return sqrt(sum);
}

__global__ void kMeansKernel(float * points, float * centroids, float * newCentroids, unsigned int * pointsPerCluster, int n, int k) {
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("\t\tThread n. %d\n", p);
	if (p < n) {
		if (p < k) {
			for (int d = 0; d < DIM; d++)
				newCentroids[d * k + p] = 0;
			pointsPerCluster[p] = 0;
		}
		int nearestId = 0;
		float minDistance = pointDistance(points, centroids, p, 0, n, k);
		float distance;
		for (int c = 1; c < k; c++) {
			distance = pointDistance(points, centroids, p, c, n, k);
			if (distance < minDistance) {
				minDistance = distance;
				nearestId = c;
			}
		}
		atomicInc(&(pointsPerCluster[nearestId]), n);
		for (int d = 0; d < DIM; d++) {
			//printf("\t\t\td = %d\n", d);
			atomicAdd(&(newCentroids[d * k + nearestId]), points[d * n + p]);
		}
		__threadfence(); // wait for pointsPerCluster
		if (p < k) {
			for (int d = 0; d < DIM; d++)
				centroids[d * k + p] = newCentroids[d * k + p] / pointsPerCluster[p];
			//printf("\t\t%u points in cluster %d\n", pointsPerCluster[p], p);
		}
	}
}

void kMeans(float * points, float * centroids, int bDim, int n, int k) {
	float * devNewCentroids;
	unsigned int * devPointsPerCluster;
	CUDA_CHECK_RETURN(cudaMalloc((void **) &devNewCentroids, k * DIM * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void **) &devPointsPerCluster, k * sizeof (unsigned int)));
	for (int i = 0; i < MAX_ITER; i++) {
		//printf("\titeration n. %d\n", i);
		kMeansKernel<<<ceil(n/(float)bDim), bDim>>>(points, centroids, devNewCentroids, devPointsPerCluster, n, k);
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
	int numTests = 1;

	for (int n = 4; n <= 65536; n *= 2) { // 4 <= n <= 65536
		printf("n = %d\n", n);

		// Allocate points memory
		points = (float *) malloc(n * DIM * sizeof(float));
		CUDA_CHECK_RETURN(cudaMalloc((void **) &devPoints, n * DIM * sizeof(float)));

		for (int k = 2; k < n; k *= 2) { // 2 <= k < n
			printf("\tk = %d\n", k);

			// Allocate centroids memory
			centroids = (float *) malloc(k * DIM * sizeof(float));
			CUDA_CHECK_RETURN(cudaMalloc((void **) &devCentroids, k * DIM * sizeof(float)));

			int minTime = -1;
			for (int bDim = 32; bDim <= 1024; bDim *= 2) { // 32 <= blockDim <= 1024
				printf("\t\tblockDim=%d\n", bDim);

				int time = 0;
				for (int i = 0; i < numTests; i++) {
					//printf("Test n. %d\n", i);

					// Generate points and centroids on host
					generatePoints(points, gen, n);
					generateCentroids(centroids, points, n, k);

					// Copy data to device
					CUDA_CHECK_RETURN(cudaMemcpy((void *) devPoints, (void *) points, n * DIM * sizeof(float), cudaMemcpyHostToDevice));
					CUDA_CHECK_RETURN(cudaMemcpy((void *) devCentroids, (void *) centroids, k * DIM * sizeof(float), cudaMemcpyHostToDevice));

					// Start timer
					auto start = std::chrono::system_clock::now();

					kMeans(devPoints, devCentroids, bDim, n, k);

					// Stop timer
					auto end = std::chrono::system_clock::now();
					auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
					time += elapsed.count();

					// Copy data back to host
					//CUDA_CHECK_RETURN(cudaMemcpy((void **) centroids, (void **) devCentroids, K * DIM * sizeof(float), cudaMemcpyDeviceToHost));
				}
				time /= numTests;
				if (time < minTime || minTime < 0)
					minTime = time;
			}
			std::cout << "\t\tTime: "<< minTime << std::endl;
			free(centroids);
			CUDA_CHECK_RETURN(cudaFree(devCentroids));
		}
		free(points);
		CUDA_CHECK_RETURN(cudaFree(devPoints));
	}
}

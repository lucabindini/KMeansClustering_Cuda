#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <iostream>
#include <random>
#include <chrono>
#include <cmath>
#include <vector>


/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 *
 * See cuda.h for error code descriptions.

#define CHECK_CUDA_RESULT(N) {											\
		CUresult result = N;												\
		if (result != 0) {													\
			printf("CUDA call on line %d returned error %d\n", __LINE__,	\
					result);													\
					exit(1);														\
		} }
*/
__global__ void helloFromGPU() {
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Hello World from GPU thread %d!\n", idx);
}



class Point {
public:
	Point(std::vector<float> coordinates) {
		this->coordinates = coordinates;
	}

	float distanceFrom(const Point p) const {
		float sum = 0;
		for (int i = 0; i < coordinates.size(); i++)
			sum += pow((coordinates.at(i) - p.getCoordinates().at(i)), 2);
		return sqrt(sum);
	}

	bool findNearest(std::vector<Point> points) {
		int minimum = 0;
		float distance = distanceFrom(points.at(0));
		float currDistance;
		bool hasChanged = false;
		for (int i = 1; i < points.size(); i++) {
			if ((currDistance = distanceFrom(points.at(i))) < distance) {
				distance = currDistance;
				minimum = i;
			}
		}
		if(minimum != centroidID)
			hasChanged = true;
		centroidID = minimum;
		return hasChanged;
	}

	const std::vector<float> &getCoordinates() const {
		return coordinates;
	}

	__device__ int getCentroidId() const {
		return centroidID;
	}

	Point(int dimension) {
		std::vector<float> coordinates = std::vector<float>();
		for (int i = 0; i < dimension; i++)
			coordinates.push_back(0);
		this->coordinates = coordinates;
	}

	Point &operator+=(const Point &rhs) {
		for (int i = 0; i < coordinates.size(); i++) {
			coordinates.at(i) += rhs.getCoordinates().at(i);
		}
		return *this;
	}

	Point &operator/=(int num) {
		for (int i = 0; i < coordinates.size(); i++) {
			coordinates.at(i) /= num;
		}
		return *this;
	}

private:
	std::vector<float> coordinates;
	int centroidID;

};

__global__ void moveCentroidKernel(int k, int n,int dimension, Point* points, Point* centroids) {
	int centroidID = threadIdx.x;
	if(centroidID < k) {
		Point newCentroid(dimension);
		int numPoints = 0;
		for (int i = 0; i < n; i++) {
			if (points[i].getCentroidId() == centroidID) {
				newCentroid += points[i];
				numPoints++;
			}
		}
		if (numPoints != 0) {
			newCentroid /= numPoints;
			centroids[centroidID] = newCentroid;
		}
	}

}

class KMeans {
public:
	KMeans(int k, int n, int dimension, int maxIter) {
		this->k = k;
		this->n = n;
		this->dimension = dimension;
		this->maxIter = maxIter;
		points = (Point *)malloc(sizeof(Point)*n);
		centroids = (Point *)malloc(sizeof(Point)*k);
		oldCentroids = (Point *)malloc(sizeof(Point)*k);
		hasChanged = false;
		init();
	}

	void execute() {
		int iter = 0;
		do {
			for(int i=0;i<k;i++)
				oldCentroids[i] = centroids[i];
			assignCentroids();
			//printAllPoints();
			//printCentroids();
			moveCentroidKernel<<<1, k>>>(k, n, dimension, points, centroids);
			iter++;
		} while (hasChanged && ((iter < maxIter) || (maxIter == -1))); //Set maxIter to -1 to remove Max Iteration criteria
	}

	void printAllPoints() {
		for (int i = 0; i < n; i++) {
			std::cout << "Point " << i << ": ";
			for (int j = 0; j < dimension; j++) {
				std::cout << points[i].getCoordinates().at(j) << " ";
			}
			std::cout << "- CentroidID: " << points[i].getCentroidId() << std::endl;
		}
		std::cout << std::endl;
	}

	void printCentroids() {
		for (int i = 0; i < k; i++) {
			std::cout << "Centroid " << i << ": ";
			for (int j = 0; j < dimension; j++) {
				std::cout << centroids[i].getCoordinates().at(j) << " ";
			}
			std::cout << std::endl;
		}
	}


private:
	int k;
	int n;
	int dimension;
	int maxIter;
	bool hasChanged;
	Point* points;
	Point* centroids;
	Point* oldCentroids;

	void init() {
		std::random_device rd;
		std::mt19937 gen(rd());
		for (int i = 0; i < n; i++) {
			std::vector<float> coordinates = std::vector<float>(dimension);
			for (int j = 0; j < dimension; j++) {
				coordinates[j] = std::generate_canonical<float, 10>(gen);
			}
			points[i]=coordinates;
		}
		for (int i = 0; i < k; i++)
			centroids[i] = points[i];
	}

	void assignCentroids() {
		hasChanged = false;
		for (int i = 0; i < n; i++) {
			if (points[i].findNearest(centroids))
				hasChanged = true;
		}
		//    if (hasChanged)
		//        std::cout << "HAS CHANGED" << std::endl;
	}


};



int main(int argc, char **argv) {
	auto start = std::chrono::system_clock::now();
	KMeans kmeans(5, 10000, 20, 10);
	kmeans.execute();
	auto end = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
	std::cout<<std::endl;
	std::cout<<"Tempo: "<< elapsed.count();
	//helloFromGPU<<<1, 10>>>();
	cudaDeviceSynchronize();
	return 0;
}


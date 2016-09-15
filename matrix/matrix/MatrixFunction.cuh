// CUDA runtime
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

typedef struct {
	int width;
	int height;
	int stride;
	float *elements;
} Matrix_;
#define IDX2C(i,j,ld) (((j)*(ld))+(i));

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
// cudaError_t multiMatriWithCuda(float *c, float *a, float *b, int widthA, int heightA, int widthB, int heightB,);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

template<int BLOCK_SIZE>
__global__ void kaisaiMatrixComputation(float *b, float *a){
	int bx = blockIdx.x;
	//int by = blockIdx.y;

	// Thread index 
	int tx = threadIdx.x;
	//int ty = threadIdx.y;

	float Csub = 0;
	// Declaration of the shared memory array as used to store the sum-matrix of A
	__shared__ float As[BLOCK_SIZE];

	As[tx] = a[bx*BLOCK_SIZE + tx];
	Csub = expf(-Lamda * As[tx]);
	b[bx*BLOCK_SIZE + tx] = Csub;
	return;
}

template<int BLOCK_SIZE>
__global__ void elementWiseDIV(float *c, float *a, float* b){
	int bx = blockIdx.x;
	//int by = blockIdx.y;

	// Thread index 
	int tx = threadIdx.x;
	//int ty = threadIdx.y;

	float Csub = 0;

	// Declaration of the shared memory array as used to store the sum-matrix of A
	__shared__ float As[BLOCK_SIZE];

	// Delcaration of the shared memory array as used to store the sub-matrix of B;
	__shared__ float Bs[BLOCK_SIZE];

	As[tx] = a[bx * BLOCK_SIZE + tx];
	Bs[tx] = b[bx * BLOCK_SIZE + tx];

	/*if (Bs[tx] >= 0 && Bs[tx] < 0.000001){
	Bs[tx] = 0.000001;
	}
	else if (Bs[tx] <= 0 && Bs[tx] > -0.000001){
	Bs[tx] = -0.000001;
	}*/

	c[bx * BLOCK_SIZE + tx] = As[tx] / Bs[tx];

	return;
}


template<int BLOCK_SIZE>
__global__ void elementWiseMUL(float *c, float *a, float* b){
	int bx = blockIdx.x;
	//int by = blockIdx.y;

	// Thread index 
	int tx = threadIdx.x;
	//int ty = threadIdx.y;

	float Csub = 0;

	// Declaration of the shared memory array as used to store the sum-matrix of A
	__shared__ float As[BLOCK_SIZE];

	// Delcaration of the shared memory array as used to store the sub-matrix of B;
	__shared__ float Bs[BLOCK_SIZE];

	As[tx] = a[bx * BLOCK_SIZE + tx];
	Bs[tx] = b[bx * BLOCK_SIZE + tx];


	c[bx * BLOCK_SIZE + tx] = As[tx] * Bs[tx];

	return;
}


template<int BLOCK_SIZE>
__global__ void distancePointToPointCUDA(float *c, float *a, float *b, int hA, int wA, int hB, int wB)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index 
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// index of the first sub-matrix of A processed by the block
	//	int a_begin = BLOCK_SIZE * bx;

	// index of the last sub-matrix of A processed by the block
	//	int a_end = a_begin +  BLOCK_SIZE - 1;

	// Step size used to iterate through the sub-matrices of A
	//	int a_step = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
	//	int b_begin =  BLOCK_SIZE * by;

	// Index of the last sub-matrix of B proceesed by the block
	//	int b_end = b_begin +  BLOCK_SIZE - 1;

	// Step size used to iterate through the sub-matrices of B
	//	int b_step = BLOCK_SIZE;

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	float Csub = 0;

	//const int B = wA;

	// Declaration of the shared memory array as used to store the sum-matrix of A
	__shared__ float As[2];

	// Delcaration of the shared memory array as used to store the sub-matrix of B;
	__shared__ float Bs[BLOCK_SIZE * 2];

	// Load the matrices from device memroy 
	// to shared memory; each thread loads 
	// one element of each matrix

#pragma unroll

	for (int i = 0; i < wA; i++){
		As[i] = a[bx * wA + i];

	}

#pragma unroll

	for (int i = 0; i < wA; i++){
		Bs[ty * wA + i] = b[by * BLOCK_SIZE * wB + ty * wB + i];
	}

	// Synchronize to make sure the matrices are loaded

	__syncthreads();

#pragma unroll
	for (int i = 0; i < wA; i++){
		float dif_ = As[i] - Bs[ty * wA + i];
		Csub += dif_ * dif_;
	}

	// Synchronized to make sure that the preceeding 
	// computation is done 

	__syncthreads();
	// Write the block sub- matrix to device memory;
	// eahc thread writes one element

	int c_line = bx;
	int c_col = by * BLOCK_SIZE + ty;
	c[c_line * hB + c_col] = Csub;
}

int distanceCompuation(int block_size, dim3 &dimsA, dim3 &dimsB, float *matrix_A, float *matrix_B, float *matrix_C, float *matrix_D);

int initCuda(){
	int devID = 0;

	cudaSetDevice(devID);

	cudaError_t error;
	cudaDeviceProp deviceProp;
	error = cudaGetDevice(&devID);

	if (error != cudaSuccess){
		printf("cudaGetDevice returned error %s (code %d), line (%d)\n", cudaGetErrorString(error), error, __LINE__);
	}

	error = cudaGetDeviceProperties(&deviceProp, devID);

	if (deviceProp.computeMode == cudaComputeModeProhibited){
		fprintf(stderr, "Error: device is runing in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
		exit(EXIT_SUCCESS);
	}

	if (error != cudaSuccess){
		printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);

	}
	else{
		printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
	}

	// Use a larger block size for Fermi and above
	int block_size = deviceProp.major < 2 ? 16 : 32;
	return block_size;
}
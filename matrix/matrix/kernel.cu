
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "setMatrix.h"
typedef struct {
	int width;
	int height;
	int stride;
	float *elements;
} Matrix_;


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
// cudaError_t multiMatriWithCuda(float *c, float *a, float *b, int widthA, int heightA, int widthB, int heightB,);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

template<int BLOCK_SIZE>
__global__ void distancePointToPointCUDA(float *c, float *a, float *b, int hA,  int wA,  int hB,  int wB)
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

	for (int i = 0; i < wA;i++){
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

	int c_line = bx ;
	int c_col =  by * BLOCK_SIZE + ty;
	c[c_line * hB + c_col] = Csub;	
}

int distanceCompuation(int block_size, dim3 &dimsA, dim3 &dimsB, float *matrix_A, float *matrix_B);

int main(int argc, char *argv[])
{


	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };
	int DIMENSIONS = 2;

	int num_samplingPoints; // 采样点的数量
	int width_originalPoints ; // 原始采样点的横向采样数量
	int height_originalPoints; // 原始采样点的纵向采样数量
	float *m_samplingPoints; // 采样点的位置；
	float *m_originalPoints; // 原始密度函数的离散点；

	num_samplingPoints = setNumSamplingPoint();
	width_originalPoints = setWidthOriginalPoint();
	height_originalPoints = setHeightOriginalPoint();

	m_samplingPoints = (float *)malloc(num_samplingPoints * 2 * sizeof(float));
	m_originalPoints = (float *)malloc(width_originalPoints * height_originalPoints * 2 * sizeof(float));

	setSamplingPoints(m_samplingPoints, num_samplingPoints, DIMENSIONS);
	setOriginalPoints(m_originalPoints, height_originalPoints, width_originalPoints, DIMENSIONS);
	

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
	
	dim3 dims_sampling_points(num_samplingPoints, 2, 1);
	dim3 dims_original_points(width_originalPoints * height_originalPoints, 2, 1);
    // Add vectors in parallel.
	
	int _result = distanceCompuation(block_size, dims_sampling_points, dims_original_points, m_samplingPoints, m_originalPoints);

	free(m_samplingPoints);
	free(m_originalPoints);
	exit(_result);

	
}

int distanceCompuation(int block_size, dim3 &dimsA, dim3 &dimsB, float *matrix_A, float *matrix_B){
	// allocate host memory for original points and sampling points
	unsigned int size_A = dimsA.x * dimsA.y * dimsA.z;
	unsigned int mem_sizeA = sizeof(float) * size_A;
	float *h_A = matrix_A;

	unsigned int size_B = dimsB.x * dimsB.y * dimsB.z;
	unsigned int mem_sizeB = sizeof(float) * size_B;
	float  *h_B = matrix_B;

	// allocate device memory 
	float *d_A, *d_B, *d_C;

	// allocate host matrix C
	dim3 dimsC(dimsA.x, dimsB.x, 1);
	unsigned int mem_sizeC = dimsC.x * dimsC.y * sizeof(float);
	float *h_C = (float *)malloc(mem_sizeC);

	if (h_C == NULL){
		fprintf(stderr, "Failed to allocate host matrix C!\n");
		exit(EXIT_FAILURE);
	}

	cudaError_t error;

	error = cudaMalloc((void**)&d_A, mem_sizeA);

	if (error != cudaSuccess){
		printf("cudaMalloc d_A returned error %s (code %d), line(%d)\n", cudaGetErrorString, error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMalloc((void**)&d_B, mem_sizeB);

	if (error != cudaSuccess){
		printf("cudaMalloc d_B returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMalloc((void**)&d_C, mem_sizeC);


	if (error != cudaSuccess){
		printf("cudaMalloc d_C returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}


	// copy host memory to device
	error = cudaMemcpy(d_A, h_A, mem_sizeA, cudaMemcpyHostToDevice);
	if (error != cudaSuccess){
		printf("cudaMemcpy (d_A, h_A) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMemcpy(d_B, h_B, mem_sizeB, cudaMemcpyHostToDevice);
	if (error != cudaSuccess){
		printf("cudaMemcpy (d_B, h_B) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	// Setup execution parameters
	block_size = 4;
	dim3 threads(1, block_size, 1);
	dim3 grid(dimsA.x, dimsB.x / block_size, 1);

	// Create and start timer
	printf("Computing result using CUDA Kernel...\n");


	cudaEvent_t start;
	error = cudaEventCreate(&start);

	if (error != cudaSuccess){
		fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	cudaEvent_t stop;
	error = cudaEventCreate(&stop);

	if (error != cudaSuccess){
		fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Record the start event
	error = cudaEventRecord(start, NULL);

	if (error != cudaSuccess){
		fprintf(stderr, " Failed to record start evern (error code %s)! \n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	
	//Performs warmup operation using distanceComputation CUDA kernel
/*
	int dA = dimsA.x;
	

	const int dims_A = 2;// *(const_cast<int*> (&dA));
	const int dims_B = 2;// *(const_cast<int*> (&dA));
	const int w_A = 2;//*(const_cast<int*> (&dA));
	const int w_B = 2;// *(const_cast<int*> (&dA));
*/
	if (block_size == 16){
		distancePointToPointCUDA<4> << <grid, threads >> >(d_C, d_A, d_B, dimsA.x, dimsA.y, dimsB.x, dimsB.y);
	}
	else{
		distancePointToPointCUDA<4> << <grid, threads >> >(d_C, d_A, d_B, dimsA.x, dimsA.y, dimsB.x, dimsB.y);
	}
	
	
	// Record the stop event
	error = cudaEventRecord(stop, NULL);

	if (error != cudaSuccess){
		fprintf(stderr, " Failed to record stop event ( error code %s)! \n", cudaGetErrorString(error));
	}

	// Wait for the stop event to complete
	error = cudaEventSynchronize(stop);

	if (error != cudaSuccess){
		fprintf(stderr, "Failed to synchronize on the stop event (error code %s)! \n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	
	float msecTotal = 0.0f;
	error = cudaEventElapsedTime(&msecTotal, start, stop);

	if (error != cudaSuccess){
		fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Copy result from device to host
	error = cudaMemcpy(h_C, d_C, mem_sizeC, cudaMemcpyDeviceToHost);

	if (error != cudaSuccess){
		printf("cudaMemcpy(h_C,d_C) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	
	for (int i = 0; i < dimsA.x; i++){
		for (int j = 0; j < dimsA.y; j++){	
			printf("%f  ", h_A[j*dimsA.x + i]);
		}
		printf("\n");
	}

	
	for (int i = 0; i < dimsB.x; i++){
		for (int j = 0; j < dimsB.y; j++){		
			printf("%f  ", h_B[ i * dimsB.y + j]);
		}
		printf("\n");
	}

	for (int i = 0; i < dimsA.x; i++){
		for (int j = 0; j < dimsB.x; j++){
			printf("%f  ", h_C[i*dimsB.x + j]);
		}
		printf("\n");
	}


	// Check  the result
	
	float *c_C = (float *)malloc(mem_sizeC);
	for (int i = 0; i < dimsA.x; i++){
		for (int j = 0; j < dimsB.x; j++){
			float diff_x;
			float diff_y;
			diff_x = *(h_A + i * dimsA.y) - *(h_B + j*dimsB.y );
			diff_y = *(h_A + i * dimsA.y + 1) - *(h_B + j*dimsB.y + 1);
			*(c_C + dimsB.x * i + j) = diff_x * diff_x + diff_y * diff_y;
		}
	}

	float diff_ = 0;
	for (int i = 0; i < dimsA.x; i++){
		for (int j = 0; j < dimsB.x; j++)
			diff_ += ( h_C[i * dimsB.x + j] - c_C[i*dimsB.x + j]) * (h_C[i * dimsB.x + j] - c_C[i*dimsB.x + j] );

	}

	printf("The difference between results of CPU and GPU is %f.\n", diff_);

	

	//Clean up memory
	//free(h_A);
	//free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n");

	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits

	cudaDeviceReset();

	return EXIT_SUCCESS;

}


// set variables for distance compuation between points
void setVariableForDistanceComputation(){




	return;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
	int *dev_d = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_d, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);



	addKernel << <1, size >> >(dev_d, dev_c, dev_b);


    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_d, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

/*
cudaError_t multiMatriWithCuda(float *c, float *a, float *b, int widthA, int heightA, int widthB, int heightB)
{
	float2 *dev_a = 0;
	float2 *dev_b = 0;
	float2 *dev_c = 0;
	
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, heightA * heightB * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, heightA * widthA * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, heightB * widthB * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}



	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, widthA * heightA * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, widthB * heightB * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_d, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
*/
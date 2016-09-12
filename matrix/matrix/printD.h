#include <iostream>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include<string.h>
void writeD(float *ptr, int num,int r){
	float* result = (float *)malloc(sizeof(float) * num);
	cudaMemcpy(result, ptr, num*sizeof(float), cudaMemcpyDeviceToHost);

	/*for (int i = 0; i < num; i++){
		std::cout <<i<<" "<< result[i] << std::endl;
	}
	std::cout << std::endl;*/

	FILE* fp;
	fp = fopen("file.txt", "at");
	if (r==1)
		fprintf(fp, "d_kasaiV\n");
	else if (r ==2)
		fprintf(fp, "d_U\n");
	else if (r==3)
		fprintf(fp, "d_kasaiU\n");
	else if (r==4)
		fprintf(fp, "d_V\n");
	else if (r == 5)
		fprintf(fp, "d_tempVectorStopCri\n");
	else if (r==6)
		fprintf(fp, "stop_valueU\n");

	for (int i = 0; i<num; i++)
	{
		fprintf(fp, "%f\n", result[i]);
	}
	fprintf(fp, "\n");
	fclose(fp);
}

void write2File(float *ptr,int num, int r){
	FILE* fp;
	if (r == 1){
		fp = fopen("point.txt", "w");
	}
	else if (r == 2){
		fp = fopen("site.txt", "w");
	}
	else if (r == 3){
		fp = fopen("originalPointsDesntiy.txt", "w");
	}
	else if (r == 4){
		fp = fopen("samplingPointsDensity.txt", "w");
	}

	for (int i = 0; i<num; i++)
	{
		fprintf(fp, "%f\n", ptr[i]);
	}
	fprintf(fp, "\n");
	fclose(fp);
}

template<typename IndexType, typename ValueType>

void printD(ValueType *ptr, IndexType num){
	ValueType* result = (ValueType *)malloc(sizeof(ValueType) * num);
	cudaMemcpy(result, ptr, num*sizeof(ValueType), cudaMemcpyDeviceToHost);

	for (int i = 0; i < num; i++){
		std::cout << i << " " << result[i] << std::endl;
	}
	std::cout << std::endl;

	/*
	FILE* fp;
	fp = fopen("file.txt", "at");
	if (r == 1)
	fprintf(fp, "d_kasaiV\n");
	else if (r == 2)
	fprintf(fp, "d_U\n");
	else if (r == 3)
	fprintf(fp, "d_kasaiU\n");
	else if (r == 4)
	fprintf(fp, "d_V\n");
	else if (r == 5)
	fprintf(fp, "d_tempVectorStopCri\n");
	else if (r == 6)
	fprintf(fp, "stop_valueU\n");

	for (int i = 0; i<num; i++)
	{
	fprintf(fp, "%d = %f\n", i, result[i]);
	}
	fprintf(fp, "\n");
	fclose(fp);
	*/
}

template<typename IndexType, typename ValueType>
void printH(ValueType *ptr, IndexType num){

	for (int i = 0; i < num; i++){
		std::cout << i << " " << ptr[i] << std::endl;
	}
	std::cout << std::endl;
}

template<typename IndexType, typename ValueType>
void printDH(ValueType *ptr1, ValueType *ptr2, IndexType num){

	ValueType* result = (ValueType *)malloc(sizeof(ValueType) * num);
	cudaMemcpy(result, ptr1, num*sizeof(ValueType), cudaMemcpyDeviceToHost);

	for (int i = 0; i < num; i++){
		std::cout << i << " " << result[i] << " " << ptr2[i] << std::endl;
	}
	std::cout << std::endl;
}
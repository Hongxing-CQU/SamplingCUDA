//#include <math>
#include <stdlib.h> 
#include <time.h>  
#define Lamda 800

int setHeightOriginalPoint(){
	int height = 64;
	return height;
}

int setWidthOriginalPoint(){
	int width = 64;
	return width;
}

int setNumSamplingPoint(){
	int num = 256;
	return num;
}

void setOriginalPoints(float *matrix, int heightMatrix, int widthMatrix, int dimensions){

	float *ip;

	for (int i = 0; i < heightMatrix; i++){
		for (int j = 0; j < widthMatrix; j++){
			ip = matrix + i * widthMatrix * dimensions + j * dimensions;
			*ip = (float)(2 * j + 1) / 2 / widthMatrix;
			ip++;
			*ip = (float)(2 * i + 1) / 2 / heightMatrix;

			//		matrix[widthMatrix * i + j ] = ( 2 * j + 1 ) / 2 / widthMatrix ;
			//	matrix[widthMatrix * i + j + widthMatrix * heightMatrix] = (2 * i + 1) / 2 / heightMatrix;
		}
	}
	return;
}

void setSamplingPoints(float *samplingPoints, int numOfPoints, int dimensions){
	srand((unsigned)time(NULL));
	for (int i = 0; i <numOfPoints; i++){
		
		for (int j = 0; j <dimensions; j++){
			*(samplingPoints + i*dimensions + j) = static_cast<float>(rand() % RAND_MAX) / RAND_MAX;
		}
	}
	return;
}

void setSamplingPointDensity(float *density, int numOfPoints){
	if (numOfPoints < 1)
		return;
	for (int i = 0; i < numOfPoints; i++){

		*(density + i) = (float)1 / numOfPoints;
	}
	return;
}


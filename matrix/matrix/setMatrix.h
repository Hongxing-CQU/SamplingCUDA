//#include <math>
#include <stdlib.h> 
#include <time.h>  

int setHeightOriginalPoint(){
	int height = 4;
	return height;
}

int setWidthOriginalPoint(){
	int width = 4;
	return width;
}

int setNumSamplingPoint(){
	int num = 2;
	return num;
}

void setOriginalPoints(float *matrix, int heightMatrix, int widthMatrix, int dimensions ){
	
	float *ip ;
	
	for (int i = 0; i < heightMatrix; i++ ){
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
	
	for (int i = 0; i < dimensions; i++){
		srand((unsigned)time(NULL));
		for (int j = 0; j < numOfPoints; j++){
			*(samplingPoints + j*dimensions + i) = j; // float(rand() / double(RAND_MAX));
		}
	}	
	return;	
}


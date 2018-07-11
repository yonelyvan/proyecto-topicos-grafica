#include <bits/stdc++.h>
using namespace std;

#define THREADS_PER_BLOCK 1024

void initData(int* M, int rows, int cols){
	for (int i=0; i<rows*cols; i++){
		//for(int j=0; j<cols; j++){
			//M[cols * i + j] = 1;
			M[i] = 1;
		//}
	}
}

void displayData(int *M, int rows, int cols){
	for (int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){
			cout<< M[cols * i + j]<<" ";
		}
		cout<<endl;
	}
}

__global__ void matrixAddKernel(int *A, int *B, int *R, int rows){
	int size = rows * rows; //Remember: square matrices
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i<size){
		R[i] = A[i] + B[i];
	}
}

void matrixAdd(int *A, int *B, int* R, int rows, int cols){
	size_t size = rows * cols * sizeof(int);
	int * d_A;
	int * d_B;
	int * d_R;
	//Allocate device memory for matrices
	cudaMalloc((void **) &d_A, size);
	cudaMalloc((void **) &d_B, size);
	cudaMalloc((void **) &d_R, size);

	//Copy B and C to device memory
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	matrixAddKernel <<< ceil((double)(size)/THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_A, d_B, d_R, rows);

	cudaMemcpy(R, d_R, size, cudaMemcpyDeviceToHost);
	//Free device matrices
	cudaFree(d_B);
	cudaFree(d_A);
	cudaFree(d_R);
}





int main(int argc, char** argv){
	//int numRows = 20;
	int rows = 100;
	int cols = 40;
	
	int nElem = rows * cols;

	int * A = (int *) malloc(nElem * sizeof(int));
	int * B = (int *) malloc(nElem * sizeof(int));
	int * R = (int *) malloc(nElem * sizeof(int));

	initData(B, rows, cols);
	initData(A, rows, cols);
	matrixAdd(A, B, R, rows, cols);
	displayData(R, rows, cols);

	free(A); free(B); free(R);
}
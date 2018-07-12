#include <bits/stdc++.h>
using namespace std;




#define THREADS_PER_BLOCK 1024//1024

void initData(int* M, int rows, int cols){
	for (int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){
			M[cols * i + j] = 2;
		}
	}
}

void displayData(int *M, int rows, int cols){
	for (int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){
			cout<< M[cols * i + j]<<" ";
		}
		cout<<endl;
	}
	cout<<endl;
}


__global__ void sum(int *a, int *b, int *r, int tam) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if( index < tam){
		r[index] = a[index] + b[index];
	}
}

void matrixAdd(int *A, int *B, int* R, int rows, int cols){
	int *d_A, *d_B, *d_R;
	int nElem = rows * cols;
	int size = nElem * sizeof(int);
	//Allocate device memory for matrices
	cudaMalloc((void **) &d_A, size);
	cudaMalloc((void **) &d_B, size);
	cudaMalloc((void **) &d_R, size);

	//Copy B and C to device memory
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	sum<<<(nElem+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_A, d_B, d_R, nElem);//run

	cudaMemcpy(R, d_R, size, cudaMemcpyDeviceToHost);
	//Free device matrices
	cudaFree(d_B);
	cudaFree(d_A);
	cudaFree(d_R);
}


int run(){
	int rows = 10;
	int cols = 10;

	int nElem = rows * cols;

	int * A = (int *) malloc(nElem * sizeof(int));
	int * B = (int *) malloc(nElem * sizeof(int));
	int * R = (int *) malloc(nElem * sizeof(int));

	initData(B, rows, cols);
	initData(A, rows, cols);
	matrixAdd(A, B, R, rows, cols);

	//displayData(A, rows, cols);
	//displayData(B, rows, cols);
	displayData(R, rows, cols);

	free(A); free(B); free(R);
}



int main(){
	run();
	return 0;
}

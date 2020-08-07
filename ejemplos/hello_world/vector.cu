// probado en cuda 10.1 agosto 2020
// nvcc suma.cu -o v && ./v
#include <bits/stdc++.h>
using namespace std;
#define THREADS_PER_BLOCK 1024 //depende de la arquitectura

//#define g 10/2
void random_ints(int *a, int tam){
   for (int i =0; i < tam; ++i){
   		a[i] = 1;//+rand()%10;
   }
}

void imprimir(int *&v, int tam){
	for(int i=0;i<tam;i++){
		cout<<v[i]<<" ";
	}
	cout<<endl;
}

//=================cuda=================
__global__ void add(int *a, int *b, int *r) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	r[index] = a[index] + b[index];
}

__global__ void sumar(int *a, int *b, int *r, int tam) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if( index < tam){
		r[index] = a[index] + b[index];
	}
}

void cuda_suma(int *a, int *b, int *r, int tam ){
	int *d_a, *d_b, *d_r; //device copies of a,b,c
	int size = tam*sizeof(int);
	//dar memoria en GPU
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_r, size);
	//copiar HOST -> DEVICE
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);//memoria device
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);//memoria device
	//run kernel //almenos debe contener un bloque
	sumar<<<(tam+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_a, d_b, d_r, tam);//run
	//copiar DEVICE -> HOST	
	cudaMemcpy(r, d_r, size, cudaMemcpyDeviceToHost);//retornar a host
	//liberar memoria
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_r);
	
}


//======================================


int main(void){
	int tam = 300*300; //250 000
	int *a, *b, *r;
	a = (int*)malloc(tam*sizeof(int));
	b = (int*)malloc(tam*sizeof(int));
	r = (int*)malloc(tam*sizeof(int));
	random_ints(a,tam);
	random_ints(b,tam);

	//suma(a,b,r,tam);
	cuda_suma(a,b,r,tam);
	//imprimir(a,tam);
	//imprimir(b,tam);
	imprimir(r,tam);
	free(a); free(b); free(r);
	return 0;
}

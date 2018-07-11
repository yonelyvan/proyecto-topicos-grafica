// nvcc suma.cu -o v && ./v
#include <bits/stdc++.h>
using namespace std;
#define THREADS_PER_BLOCK 1024 //depende de la arquitectura
//#define THREADS_PER_BLOCK 16
#define threadsPB 8

void random_ints(int **&M, int rows, int cols){
    for (int i =0; i < rows; ++i){
        for (int j =0; j < cols; ++j){
            M[i][j] = 1+rand()%10;
        }
    }
}

void resize(int **&M,int rows, int cols){
    M = (int **) malloc(rows * sizeof(int*)) ;
    for(int i = 0; i<rows; i++) {
        M[i] = (int *) malloc(cols * sizeof(int));
    }
}


void imprimir(int **&M, int rows, int cols){
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            cout<<M[i][j]<<" ";
        }
        cout<<endl;
    }
    cout<<endl;
}

//void createMatrixHostCUDA(int**& host, int**& device, int **& aux, int size, int rows, int cols ){
void createMatrixHostCUDA(int**& device, int rows, int cols ){
    //aux =(int **)malloc(rows*sizeof(int*));

    //cudaMalloc((void **)&aux[0],size);
    cudaMalloc((void **)&device,rows*sizeof(int*));

    //for (int i=1; i<rows;++i){
      //  aux[i]=aux[i-1]+cols;
    //}
    //cudaMemcpy(device, aux, rows*sizeof(int*), cudaMemcpyHostToDevice);
}

//=================cuda=================
__global__ void sum(int **A, int **B, int **R, int rows, int cols){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<rows && j<cols){
        R[i][j] = A[i][j] + B[i][j];
    }
}



void cuda_suma(int **h_A, int **h_B, int **h_R, int rows, int cols ){
    int **d_A, **d_B, **d_R; //device copias
    //int **a_aux, **b_aux, **c_aux;
    int size = rows * cols * sizeof(int*);
    //dar memoria en GPU
    //createMatrixHostCUDA(h_A,d_A,a_aux,size,rows,cols);
    //createMatrixHostCUDA(h_B,d_B,b_aux,size,rows,cols);
    //createMatrixHostCUDA(h_R,d_R,c_aux,size,rows,cols);
    createMatrixHostCUDA(d_A,rows,cols);
    createMatrixHostCUDA(d_B,rows,cols);
    createMatrixHostCUDA(d_R,rows,cols);
    //copiar HOST -> DEVICE
    //cudaMemcpy(a_aux[0], h_A[0], size, cudaMemcpyHostToDevice);
    //cudaMemcpy(b_aux[0], h_B[0], size, cudaMemcpyHostToDevice);
    //run kernel //almenos debe contener un bloque
    dim3 threadPerBlock(threadsPB, threadsPB);
    dim3 blockPerGrid((rows+threadPerBlock.x-1)/threadPerBlock.x,(cols+threadPerBlock.y-1)/threadPerBlock.y);
    sum<<<blockPerGrid,threadPerBlock>>>(d_A,d_B,d_R,rows,cols);
    //sum<<<1,512>>>(d_A,d_B,d_R,rows,cols);

    //=====

    //=====

    //copiar DEVICE -> HOST 
    cudaMemcpy(h_R,d_R, size, cudaMemcpyDeviceToHost);

    //free(h_A); free(h_B); free(h_R);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_R);
    //cudaFree(a_aux); cudaFree(b_aux);cudaFree(c_aux);
}

//======================================

int main(){
    int rows = 8;
    int cols = 8;
    int **A, **B, **R;
    
    resize(A,rows,cols);
    resize(B,rows,cols);
    resize(R,rows,cols);

    random_ints(A,rows,cols);
    random_ints(B,rows,cols);

    cuda_suma(A,B,R,rows,cols);
    imprimir(A,rows,cols);
    imprimir(B,rows,cols);
    imprimir(R,rows,cols);
}



/*
int main(){
    int rows=row;
    int cols=column;
    //srand (time(NULL));
    
    int **h_A, **h_B, **h_R;
    int **d_A, **d_B, **d_R;
    int **a_aux, **b_aux, **c_aux;
    int size = row* column * sizeof(int*);

    
    createMatrixHostCUDA(h_A,d_A,a_aux,size,row,column);
    createMatrixHostCUDA(h_B,d_B,b_aux,size,row,column);
    createMatrixHostCUDA(h_R,d_R,c_aux,size,row,column);
    
    random_ints(h_A,rows,cols);
    random_ints(h_B,rows,cols);

    cudaMemcpy(a_aux[0], h_A[0], size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_aux[0], h_B[0], size, cudaMemcpyHostToDevice);

    
    dim3 threadPerBlock(threadsPB, threadsPB);
    dim3 blockPerGrid((rows+threadPerBlock.x-1)/threadPerBlock.x,(cols+threadPerBlock.y-1)/threadPerBlock.y);
        
    //scalarMult<<<blockPerGrid,threadPerBlock>>>(d_A,2,d_R);
    Multi<<<blockPerGrid,threadPerBlock>>>(d_A,d_B,d_R);
    cudaMemcpy(h_R[0],c_aux[0], size, cudaMemcpyDeviceToHost);
    
    print(h_A,rows,cols);
    print(h_B,rows,cols);
    print(h_R,rows,cols);

    free(h_A); free(h_B); free(h_R);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_R);
    cudaFree(a_aux[0]);cudaFree(c_aux[0]);
    return 0;
}*/
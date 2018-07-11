#include <iostream>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <math.h>

using namespace std;
//4000x4000 8
#define row 50
#define column 65

#define THREADS_PER_BLOCK 1024//64//1024//8

// Funcion para generar numeros randoms en mi matrix:
void randomsInt(int **& matrix){
    for(int i=0;i<row;++i){
      for(int j=0;j<column;++j){
            matrix[i][j] = 1; //rand()% 2 + 1;;
        }
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

void resize(int **&M,int rows, int cols){
    M = (int **) malloc(rows * sizeof(int*)) ;
    for(int i = 0; i<rows; i++) {
        M[i] = (int *) malloc(cols * sizeof(int));
    }
}

void resize_matrix(int**& host, int rows, int cols ){
    int size = rows* cols * sizeof(int*);
    host = (int **)malloc(rows*sizeof(int*));
    host[0]=(int *)malloc(size);
    for (int i=1; i<rows;++i){
        host[i]=host[i-1]+cols;
    }
}

// Funcion imprimir:
void print(int ** a){
    for(int i=0;i<row;++i){
        for(int j=0;j<column;++j){
            cout<<a[i][j]<<'\t';
        }       
    cout<<endl;
    }

    cout<<endl;
}


// =====================================================================
void createMatrixCUDA(int**& device, int **& aux, int rows, int cols){
    int size = rows* cols* sizeof(int*);
    aux =(int **)malloc(rows*sizeof(int*));
    cudaMalloc((void **)&aux[0],size);
    cudaMalloc((void **)&device,rows*sizeof(int*));
    for (int i=1; i<rows;++i){
        aux[i]=aux[i-1]+cols;
    }
    cudaMemcpy(device, aux, rows*sizeof(int*), cudaMemcpyHostToDevice);
}

__global__ void sum(int **A, int **B, int **C){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<row && j<column){
        C[i][j] =  A[i][j] + B[i][j];
    }
}


void suma_cuda(int **a, int **b, int **c, int rows, int cols){
    int **d_a, **d_b, **d_c;
    int **a_aux, **b_aux, **c_aux;
    int size = row* column * sizeof(int*);

    createMatrixCUDA(d_a,a_aux,row,column);
    createMatrixCUDA(d_b,b_aux,row,column);
    createMatrixCUDA(d_c,c_aux,row,column);

    cudaMemcpy(a_aux[0], a[0], size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_aux[0], b[0], size, cudaMemcpyHostToDevice);

    dim3 threadPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blockPerGrid((row+threadPerBlock.x-1)/threadPerBlock.x, (column+threadPerBlock.y-1)/threadPerBlock.y);

    sum<<<blockPerGrid,threadPerBlock>>>(d_a,d_b,d_c);
    //sum<<<(rows*cols+threadsPB-1)/threadsPB,threadsPB>>>(d_a, d_b, d_c);//run
    

    cudaMemcpy(c[0],c_aux[0], size, cudaMemcpyDeviceToHost);
    cudaFree(d_a);cudaFree(d_c);
    cudaFree(a_aux); cudaFree(b_aux); cudaFree(c_aux);
    //cudaFree(a_aux[0]);cudaFree(c_aux[0]);
}
// =====================================================================


int main(){
    int rows=row;
    int cols=column;
    int **a, **b, **c;
    resize_matrix(a,rows,cols);
    resize_matrix(b,rows,cols);
    resize_matrix(c,rows,cols);
    randomsInt(a);
    randomsInt(b);
    //imagebn a, imagen b, imagen c (fx)
    suma_cuda(a,b,c,rows,cols);
    imprimir(a,rows,cols);
    imprimir(b,rows,cols);
    imprimir(c,rows,cols);
    free(a); free(b); free(c);

    return 0;
}


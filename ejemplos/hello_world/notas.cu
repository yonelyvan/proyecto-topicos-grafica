#include <cuda.h>
#include <stdio.h>
#include <math.h>

#define THREADS_PER_BLOCK 1024

#define N 2

__global__ void add2D(int **d_x, int **d_y, int **d_sum){
    d_sum[threadIdx.y][threadIdx.x]=d_x[threadIdx.y][threadIdx.x]+d_y[threadIdx.y][threadIdx.x];
    //cuPrintf("d_sum[%d][%d]=%d\n",threadIdx.y, threadIdx.x, d_sum[threadIdx.y][threadIdx.x]);   
}

__global__ void sum(int **A, int **B, int **R, int rows, int cols){
    int i = blockIdx.y * blockDim.y + threadIdx.y;//filas
    int j = blockIdx.x * blockDim.x + threadIdx.x;//columnas
    if(i<rows && j<cols){
        R[i][j] = A[i][j] + B[i][j];
    }
}

int main(){
    int x[N][N], y[N][N], sum[N][N];
    int **d_x, **d_y, **d_sum;
    size_t dpitch_x=N, dpitch_y=N, dpitch_sum=N;
//  size_t dpitch;
    dim3 blockDim(N,N,1);
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            x[i][j]=3;
            y[i][j]=6;
        }
    }
    


    cudaMallocPitch((void**)&d_x, &dpitch_x, N*sizeof(int), N);
    cudaMallocPitch((void**)&d_y, &dpitch_y, N*sizeof(int), N);
    cudaMemcpy2D(d_x,dpitch_x, x, N*sizeof(int), N*sizeof(int),N,cudaMemcpyHostToDevice);
    
    //add2D<<<1,blockDim>>>(d_x,d_y,d_sum);
    sum<<<(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_x,d_y,d_sum,N,N);
    cudaMemcpy2D(sum,N*sizeof(int), d_sum, dpitch_sum, N*sizeof(int), N, cudaMemcpyDeviceToHost);
  for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            printf("%d ",sum[i][j]);
        }
        printf("\n");
    }
    return 0;
}

///////////////////

#include <bits/stdc++.h>
using namespace std;
#define N 10
#define THREADS_PER_BLOCK 1024 


void random_ints(int **&M, int rows, int cols){
    for (int i =0; i < rows; ++i){
        for (int j =0; j < cols; ++j){
            M[i][j] = 1;
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

__global__ void add2D(int **d_x, int **d_y, int **d_sum){
    d_sum[threadIdx.y][threadIdx.x]=d_x[threadIdx.y][threadIdx.x]+d_y[threadIdx.y][threadIdx.x];
    printf("ok %s\n",1 );
    //cuPrintf("d_sum[%d][%d]=%d\n",threadIdx.y, threadIdx.x, d_sum[threadIdx.y][threadIdx.x]);   
}

__global__ void suma(int **A, int **B, int **C){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<N && j<N){
        C[i][j] = 5;//A[i][j] + B[i][j];
    }
}


int main(){
    int **x, **y, **sum;
    resize(x,N,N);
    resize(y,N,N);
    resize(sum,N,N);
    random_ints(x,N,N);
    random_ints(y,N,N);
    random_ints(sum,N,N);


    int **d_x, **d_y, **d_sum;
    size_t dpitch_x=N, dpitch_y=N, dpitch_sum=N;

    dim3 blockDim(N,N,1);
    
    //cudaPrintfInit();

    cudaMallocPitch((void**)&d_x, &dpitch_x, N*sizeof(int), N);
    cudaMallocPitch((void**)&d_y, &dpitch_y, N*sizeof(int), N);
    cudaMemcpy2D(d_x,dpitch_x, x, N*sizeof(int), N*sizeof(int),N,cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_y,dpitch_y, y, N*sizeof(int), N*sizeof(int),N,cudaMemcpyHostToDevice);
    
    add2D<<<1,blockDim>>>(d_x,d_y,d_sum);
    dim3 threadPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    //rows, cols
    dim3 blockPerGrid((N+threadPerBlock.x-1)/threadPerBlock.x,(N+threadPerBlock.y-1)/threadPerBlock.y);

    //suma<<<(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_x,d_y,d_sum);
    //suma<<<blockPerGrid,threadPerBlock>>>(d_x,d_y,d_sum);
    //cudaPrintfDisplay(stdout,true);
    //cudaPrintfEnd();
    cudaMemcpy2D(sum,N*sizeof(int), d_sum, dpitch_sum, N*sizeof(int), N, cudaMemcpyDeviceToHost);

    imprimir(x,N,N);
    imprimir(y,N,N);
    imprimir(sum,N,N);
    free(x); free(y); free(sum);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_sum);
    return 0;
}


///////////////////////
#include <stdio.h>
//#include <cutil_inline.h>

#define BLOCK_SIZE 16

__global__ static void AddKernel(int *d_Buff1, int *d_Buff2, int *d_Buff3, size_t pitch, int iMatSizeM, int iMatSizeN){
    const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int index = pitch/sizeof(int);
    if(tidx<iMatSizeM && tidy<iMatSizeN){
        d_Buff3[tidx * index  + tidy] = d_Buff1[tidx * index + tidy] + d_Buff2[tidx * index + tidy];
    }

}

void printMatrix(int *pflMat, int iMatSizeM, int iMatSizeN){
    for(int idxM = 0; idxM < iMatSizeM; idxM++){
        for(int idxN = 0; idxN < iMatSizeN; idxN++){
            printf("%d\t",pflMat[(idxM * iMatSizeN) + idxN]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(){
      int iMatSizeM=0,iMatSizeN=0;
      printf("Enter size of Matrix(M*N):");
      scanf("%d %d",&iMatSizeM,&iMatSizeN);

      int *h_flMat1 = (int*)malloc(iMatSizeM * iMatSizeN * sizeof(int));
      int *h_flMat2 = (int*)malloc(iMatSizeM * iMatSizeN * sizeof(int));
      int *h_flMatSum = (int*)malloc(iMatSizeM * iMatSizeN * sizeof(int));

      for(int j=0;j<(iMatSizeM*iMatSizeN);j++){
            h_flMat1[j]=1;
            h_flMat2[j]=1;
      }

      printf("Matrix 1\n");
      printMatrix(h_flMat1, iMatSizeM, iMatSizeN);
      printf("Matrix 2\n");
      printMatrix(h_flMat2, iMatSizeM, iMatSizeN);

      int *d_flMat1, *d_flMat2, *d_flMatSum;;

      size_t d_MatPitch;

      cudaMallocPitch((void**)&d_flMat1,&d_MatPitch,iMatSizeN*sizeof(int),iMatSizeM);

      cudaMallocPitch((void**)&d_flMat2,&d_MatPitch,iMatSizeN*sizeof(int),iMatSizeM);

      cudaMallocPitch((void**)&d_flMatSum,&d_MatPitch,iMatSizeN*sizeof(int),iMatSizeM);

      cudaMemcpy2D(d_flMat1,d_MatPitch,h_flMat1,iMatSizeN * sizeof(int), iMatSizeN * sizeof(int), iMatSizeM, cudaMemcpyHostToDevice);
      cudaMemcpy2D(d_flMat2,d_MatPitch,h_flMat2,iMatSizeN * sizeof(int), iMatSizeN * sizeof(int), iMatSizeM, cudaMemcpyHostToDevice);
      dim3 blocks(1,1,1);
      dim3 threadsperblock(BLOCK_SIZE,BLOCK_SIZE,1);
      blocks.x=((iMatSizeM/BLOCK_SIZE) + (((iMatSizeM)%BLOCK_SIZE)==0?0:1));
      blocks.y=((iMatSizeN/BLOCK_SIZE) + (((iMatSizeN)%BLOCK_SIZE)==0?0:1));

      AddKernel<<<blocks, threadsperblock>>>(d_flMat1, d_flMat2, d_flMatSum, d_MatPitch, iMatSizeM,iMatSizeN);

      cudaThreadSynchronize();

      cudaMemcpy2D(h_flMatSum, iMatSizeN * sizeof(int),d_flMatSum, d_MatPitch, iMatSizeN * sizeof(int), iMatSizeM, cudaMemcpyDeviceToHost);

      cudaFree(d_flMat1);
      cudaFree(d_flMat2);
      cudaFree(d_flMatSum);

      printf("Matrix Sum\n");
      printMatrix(h_flMatSum, iMatSizeM, iMatSizeN);
}

//===============================================

#include <bits/stdc++.h>
using namespace std;

#define BLOCK_SIZE 16

__global__ static void AddKernel(int *d_Buff1, int *d_Buff2, int *d_Buff3, size_t pitch, int iMatSizeM, int iMatSizeN){
    const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int index = pitch/sizeof(int);
    if(tidx<iMatSizeM && tidy<iMatSizeN){
        d_Buff3[tidx * index  + tidy] = d_Buff1[tidx * index + tidy] + d_Buff2[tidx * index + tidy];
    }

}

void printMatrix(int *pflMat, int iMatSizeM, int iMatSizeN){
    for(int idxM = 0; idxM < iMatSizeM; idxM++){
        for(int idxN = 0; idxN < iMatSizeN; idxN++){
            printf("%d\t",pflMat[(idxM * iMatSizeN) + idxN]);
        }
        printf("\n");
    }
    printf("\n");
}




int main(){
      int iMatSizeM=10,iMatSizeN=10;

      int *h_flMat1 = (int*)malloc(iMatSizeM * iMatSizeN * sizeof(int));
      int *h_flMat2 = (int*)malloc(iMatSizeM * iMatSizeN * sizeof(int));
      int *h_flMatSum = (int*)malloc(iMatSizeM * iMatSizeN * sizeof(int));

      for(int j=0;j<(iMatSizeM*iMatSizeN);j++){
            h_flMat1[j]=1;
            h_flMat2[j]=1;
      }

      printf("Matrix 1\n");
      printMatrix(h_flMat1, iMatSizeM, iMatSizeN);
      printf("Matrix 2\n");
      printMatrix(h_flMat2, iMatSizeM, iMatSizeN);

      int *d_flMat1, *d_flMat2, *d_flMatSum;;

      size_t d_MatPitch;

      cudaMallocPitch((void**)&d_flMat1,&d_MatPitch,iMatSizeN*sizeof(int),iMatSizeM);
      cudaMallocPitch((void**)&d_flMat2,&d_MatPitch,iMatSizeN*sizeof(int),iMatSizeM);
      cudaMallocPitch((void**)&d_flMatSum,&d_MatPitch,iMatSizeN*sizeof(int),iMatSizeM);

      cudaMemcpy2D(d_flMat1,d_MatPitch,h_flMat1,iMatSizeN * sizeof(int), iMatSizeN * sizeof(int), iMatSizeM, cudaMemcpyHostToDevice);
      cudaMemcpy2D(d_flMat2,d_MatPitch,h_flMat2,iMatSizeN * sizeof(int), iMatSizeN * sizeof(int), iMatSizeM, cudaMemcpyHostToDevice);
      dim3 blocks(1,1,1);
      dim3 threadsperblock(BLOCK_SIZE,BLOCK_SIZE,1);
      blocks.x=((iMatSizeM/BLOCK_SIZE) + (((iMatSizeM)%BLOCK_SIZE)==0?0:1));
      blocks.y=((iMatSizeN/BLOCK_SIZE) + (((iMatSizeN)%BLOCK_SIZE)==0?0:1));

      AddKernel<<<blocks, threadsperblock>>>(d_flMat1, d_flMat2, d_flMatSum, d_MatPitch, iMatSizeM,iMatSizeN);

      cudaThreadSynchronize();

      cudaMemcpy2D(h_flMatSum, iMatSizeN * sizeof(int),d_flMatSum, d_MatPitch, iMatSizeN * sizeof(int), iMatSizeM, cudaMemcpyDeviceToHost);

      cudaFree(d_flMat1);
      cudaFree(d_flMat2);
      cudaFree(d_flMatSum);

      printf("Matrix Sum\n");
      printMatrix(h_flMatSum, iMatSizeM, iMatSizeN);
}


//========start_tiempo:
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

 
    cudaEventRecord(start,0);
    sum<<<blockPerGrid,threadPerBlock>>>(d_a,d_b,d_c);
    cudaEventRecord(end,0);
    cudaEventSynchronize(end);
    float elapsedTime;

    cudaEventElapsedTime(&elapsedTime,start,end);
    cout<<"El tiempo es:   "<<elapsedTime<<endl;
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    //========tiempo:


//====================================================================================================
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




////////////////////////
//Kernel function (point c of exercise)
__global__ void matrixPerRowsAddKernel(int *A, int *B, int *C, int nRows){
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i<nRows){
    for (int j=0; j<nRows; j++){
      A[i * nRows + j] = B[i * nRows + j] + C[i * nRows + j];
    }
  }
}

//Kernel function (point d of exercise)
__global__ void matrixPerColumnsAddKernel(int *A, int *B, int *C, int nColumns){
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i<nColumns){
    for (int j=0; j<nColumns; j++){
      A[i + j * nColumns] = B[i + j * nColumns] + C[i + j * nColumns];
    }
  }
}
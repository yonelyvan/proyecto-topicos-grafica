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
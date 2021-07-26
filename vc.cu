#include <iostream>
#include <vector>

__global__ void mul(int *a, int *b, int *c, int N) {
  // Global index
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;


  for (int i = 0; i < N; i++) {
    int res = a[N*row+i] * b[N*i+col];
    c[N*row+col] += res;
  }
}

//Make  code  callable  from  Chapel
extern "C" {
  void mulCUDA(int *A, int *B, int *C, int start, int end, int GPUN) {

  double BLOCKSIZE = 32;
  int GRID = ceil(GPUN/BLOCKSIZE);

  int *da, *db, *dc; // Device  variables

  cudaMalloc(&da, sizeof(int) * GPUN*GPUN);
  cudaMalloc(&db, sizeof(int) * GPUN*GPUN);
  cudaMalloc(&dc, sizeof(int) * GPUN*GPUN);

  //copy data to gpu
  cudaMemcpy(da, A, sizeof(int) * GPUN*GPUN, cudaMemcpyHostToDevice);
  cudaMemcpy(db, B, sizeof(int) * GPUN*GPUN, cudaMemcpyHostToDevice);

  //Kernel
  mul<<<grid, block>>>(da, db, dc, GPUN);

  // Copy back to host
  cudaMemcpy(C, dc, sizeof(int)* GPUN*GPUN, cudaMemcpyDeviceToHost);

  // Free gpu
  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);

  }
}

#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define KERNEL_SIZE   3
#define KERNEL_RADIUS 1

#define TILE_SIZE     KERNEL_SIZE
#define CACHE_SIZE    (KERNEL_SIZE + (KERNEL_RADIUS * 2))

__constant__ float deviceKernel[KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE];

__global__ void conv3d(float *input, float *output,
                        const int z_size, const int y_size, const int x_size) {

  int blockX = blockIdx.x * TILE_SIZE;
  int threadX = threadIdx.x;
  int blockY = blockIdx.y * TILE_SIZE;
  int threadY = threadIdx.y;
  int blockZ = blockIdx.z * TILE_SIZE;
  int threadZ = threadIdx.z;

  __shared__ float tileCache[CACHE_SIZE][CACHE_SIZE][CACHE_SIZE];

  int tileIdx = threadZ * (KERNEL_SIZE * KERNEL_SIZE) + threadY * KERNEL_SIZE + threadX;

  int tileX = tileIdx % CACHE_SIZE;
  int tileY = (tileIdx / CACHE_SIZE) % CACHE_SIZE;
  int inputX = blockX + tileX - KERNEL_RADIUS;
  int inputY = blockY + tileY - KERNEL_RADIUS;
  int inputZPartial = blockZ - KERNEL_RADIUS;
  int inputZ;

  for (int i = 0; i < CACHE_SIZE; i += 1) {
    inputZ = inputZPartial + i;

    if (inputX >= 0 && inputX < x_size && inputY >= 0 && inputY < y_size && inputZ >= 0 && inputZ < z_size) {
      tileCache[tileX][tileY][i] = input[inputZ * (y_size * x_size) + inputY * x_size + inputX];
    } else {
      tileCache[tileX][tileY][i] = 0;
    }
  }

  __syncthreads();

  int xPos = blockX + threadX;
  int yPos = blockY + threadY;
  int zPos = blockZ + threadZ;

  if (xPos >= 0 && xPos < x_size && yPos >= 0 && yPos < y_size && zPos >= 0 && zPos < z_size) {
    float result = 0;

    for (int x = 0; x < KERNEL_SIZE; x += 1) {
      for (int y = 0; y < KERNEL_SIZE; y += 1) {
        for (int z = 0; z < KERNEL_SIZE; z += 1) {
          result +=
            tileCache[threadX + x][threadY + y][threadZ + z] *
            deviceKernel[z * (KERNEL_SIZE * KERNEL_SIZE) + y * KERNEL_SIZE + x];
        }
      }
    }
    
    output[zPos * (y_size * x_size) + yPos * x_size + xPos] = result;
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  cudaMalloc((void **)&deviceInput, inputLength * sizeof(float));
  cudaMalloc((void **)&deviceOutput, inputLength * sizeof(float));
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU her
  int inputsize = z_size*y_size*x_size;
  cudaMemcpy(deviceInput, hostInput + 3, inputsize * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(deviceKernel, hostKernel, kernelLength * sizeof(float));
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 dimBlock(TILE_SIZE, TILE_SIZE, TILE_SIZE);
  dim3 dimGrid(ceil(x_size/double(TILE_SIZE)), ceil(y_size/double(TILE_SIZE)), ceil(z_size/double(TILE_SIZE)));
  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  cudaMemcpy(hostOutput + 3, deviceOutput, inputsize * sizeof(float), cudaMemcpyDeviceToHost);
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

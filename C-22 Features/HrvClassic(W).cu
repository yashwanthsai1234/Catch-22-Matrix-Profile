#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// Function to calculate differences between consecutive elements
void diff(const double a[], const int size, double b[]) {
	for (int i = 1; i < size; i++) {
		b[i - 1] = a[i] - a[i - 1];
	}
}

// CUDA kernel to calculate differences between consecutive elements
__global__ void cudaDiffKernel(const double* y, int size, double* Dy) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < size - 1) {
		Dy[tid] = y[tid + 1] - y[tid];	
	}
}

// CUDA kernel to calculate PNN40
__global__ void calculatePNN40Kernel(double* Dy, int size, double pNNx, double* pnn40) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < size - 1) {
		pnn40[tid] = (fabs(Dy[tid]) * 1000 > pNNx) ? 1.0 : 0.0;
	}
	printf("Thread %d: Dy[%d] = %f, pnn40[%d] = %f\n", tid, tid, Dy[tid], tid, pnn40[tid]);
}

// CUDA kernel for reduction
__global__ void reductionKernel(double* pnn40, int size, double* result) {	
	extern __shared__ double shared_data[];
	int tid = threadIdx.x;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	shared_data[tid] = (idx < size) ? pnn40[idx] : 0.0;
	__syncthreads();
	// Parallel reduction to calculate the sum of 1s
	for (int stride = 1; stride < blockDim.x; stride *= 2) {
		int index = 2 * stride * tid;
		if (index < blockDim.x) {
			shared_data[index] += shared_data[index + stride];
		}
	__syncthreads();
	}
	if (tid == 0) {
		atomicAdd(reinterpret_cast<unsigned long long*>(result),
		__double_as_longlong(shared_data[0]));
	}
}
double MD_hrv_classic_pnn40(const double y[], const int size) {
	const int numThreadsPerBlock = 256;
	const int numBlocksPerGrid = (size + numThreadsPerBlock - 1) / numThreadsPerBlock;
	const double pNNx = 40.0;
	double* d_y, *d_Dy;
	cudaMalloc((void**)&d_y, size * sizeof(double));
	cudaMalloc((void**)&d_Dy, (size - 1) * sizeof(double));
	cudaMemcpy(d_y, y, size * sizeof(double), cudaMemcpyHostToDevice);
	cudaDiffKernel<<<numBlocksPerGrid, numThreadsPerBlock>>>(d_y, size, d_Dy);
	cudaDeviceSynchronize();
	double* d_pnn40, h_pnn40 = 0.0;
	cudaMalloc((void**)&d_pnn40, (size - 1) * sizeof(double));
	calculatePNN40Kernel<<<numBlocksPerGrid, numThreadsPerBlock>>>(d_Dy, size, pNNx, d_pnn40);
	double* d_h_pnn40;
	cudaMalloc((void**)&d_h_pnn40, sizeof(double));

	// reductionKernel<<<1, numThreadsPerBlock, numThreadsPerBlock * sizeof(double)>>>(d_pnn40, size - 1, &h_pnn40);
	// Call the reduction kernel
	reductionKernel<<<1, numThreadsPerBlock, numThreadsPerBlock * sizeof(double)>>>(d_pnn40, size - 1, d_h_pnn40);
	// Copy the result back from device to host
	cudaMemcpy(&h_pnn40, d_h_pnn40, sizeof(double), cudaMemcpyDeviceToHost);
	// Print the result
	double result;
	result = h_pnn40/(size-1);
	printf("Final Result from Host: %f\n", h_pnn40);
	printf("Result: %f\n", result);
	// Copy data back from device to host
	cudaMemcpy(&h_pnn40, d_pnn40, sizeof(double), cudaMemcpyDeviceToHost);
	// Free device memory
	cudaFree(d_h_pnn40);
	cudaFree(d_y);
	cudaFree(d_Dy);
	cudaFree(d_pnn40);
	// free(h_Dy);
	return h_pnn40 / (size - 1);
}

int main()
{
	const int array_size = 15000;
        double* y = (double*)malloc(array_size * sizeof(double));
	int size = 0;
        double value = 0;

        const char* file_path = "/mnt/c/Users/yashw/Downloads/test.txt";
	FILE* infile = fopen(file_path, "r");
	if (infile == NULL) {
		perror("Error opening file");
		return ;
	}
	printf("File Data:\n");
	printf("Top 20 Values:\n");
	for (int i = 0; i < 20 && fscanf(infile, "%lf", &value) == 1; ++i) {
		y[size++] = value;
		printf("%lf\n", value);  // Print each value to the console
	}

	while (size < array_size && fscanf(infile, "%lf", &value) == 1) {
		y[size++] = value;
	}
	fclose(infile);
	double result = MD_hrv_classic_pnn40(y, size);
	// printf("Result: %f\n", result);
	free(y);
	return 0;
}
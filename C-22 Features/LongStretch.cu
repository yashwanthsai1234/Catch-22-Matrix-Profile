#include <cuda_runtime.h>
#include <stdio.h>
__global__ void mean_kernel(const double* d_input, double* d_partial_sums, int size) {
	extern __shared__ double s__data[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	s__data[tid] = (i < size) ? d_input[i] : 0;
	__syncthreads();
	for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			s__data[tid] += s__data[tid + s];
		}
		__syncthreads();
	}
	if (tid == 0) d_partial_sums[blockIdx.x] = s__data[0];
}
double compute_mean_cuda(const double* h_input, int size) {
	double *d_input, *d_partial_sums, *h_partial_sums;
	int threadsPerBlock = 256;
	int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
	int partialSumsSize = blocks * sizeof(double);
	cudaMalloc(&d_input, size * sizeof(double));
	cudaMalloc(&d_partial_sums, partialSumsSize);
	h_partial_sums = (double*)malloc(partialSumsSize);
	cudaMemcpy(d_input, h_input, size * sizeof(double), cudaMemcpyHostToDevice);
	mean_kernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(d_input, d_partial_sums, size);
	cudaDeviceSynchronize();
	cudaMemcpy(h_partial_sums, d_partial_sums, partialSumsSize, cudaMemcpyDeviceToHost);
	double totalSum = 0;
	for (int i = 0; i < blocks; i++) {
		totalSum += h_partial_sums[i];
	}
	cudaFree(d_input);
	cudaFree(d_partial_sums);
	free(h_partial_sums);
	return totalSum / size;
}

__global__ void countZeroStretches(const int *yBin, int size, int *blockResults, int *blockStarts, int *blockEnds) {
	extern __shared__ int sdata[];
	int tid = threadIdx.x;
	int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = (globalIdx < size) ? yBin[globalIdx] : 1;
	__syncthreads();
	int count = 0, maxCount = 0;
	for (int i = 0; i < blockDim.x; i++) {
		if (sdata[i] == 0) {
			count++;
			maxCount = max(maxCount, count);
		} else {
			count = 0;
		}
	}
	if (tid == 0) {
		blockResults[blockIdx.x] = maxCount;
	}
	if (tid == 0) {
		int startCount = 0;
		while (startCount < blockDim.x && sdata[startCount] == 0) {
			startCount++;
		}
		blockStarts[blockIdx.x] = startCount;
		int endCount = 0, i = blockDim.x - 1;
		while (i >= 0 && sdata[i] == 0) {
			endCount++;
			i--;
		}
		blockEnds[blockIdx.x] = endCount;
	}
}


__global__ void binarizeDiffs(const double *y, int *yBin, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size - 1) {
		double diffTemp = y[i + 1] - y[i];
		yBin[i] = diffTemp < 0 ? 0 : 1;
	}
}
double SB_BinaryStats_diff_longstretch0_CUDA(const double y[], int size) {
	double *d_y;
	int *d_yBin, *d_blockResults, *d_blockStarts, *d_blockEnds;
	size_t bytes = size * sizeof(double);
	size_t bytesBin = (size - 1) * sizeof(int);
	cudaMalloc(&d_y, bytes);
	cudaMalloc(&d_yBin, bytesBin);
	int numBlocks = (size + 255) / 256;
	cudaMalloc(&d_blockResults, numBlocks * sizeof(int));
	cudaMalloc(&d_blockStarts, numBlocks * sizeof(int));
	cudaMalloc(&d_blockEnds, numBlocks * sizeof(int));
	cudaMemcpy(d_y, y, bytes, cudaMemcpyHostToDevice);
	binarizeDiffs<<<numBlocks, 256>>>(d_y, d_yBin, size);
	countZeroStretches<<<numBlocks, 256, 256 * sizeof(int)>>>(d_yBin, size - 1, d_blockResults, d_blockStarts, d_blockEnds);
	int *blockResults = (int *)malloc(numBlocks * sizeof(int));
	int *blockStarts = (int *)malloc(numBlocks * sizeof(int));
	int *blockEnds = (int *)malloc(numBlocks * sizeof(int));
	cudaMemcpy(blockResults, d_blockResults, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(blockStarts, d_blockStarts, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(blockEnds, d_blockEnds, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
	printf("Block Results:\n");
	for (int i = 0; i < numBlocks; ++i) {
		printf("Block %d: Max Stretch = %d, Start Stretch = %d, End Stretch = %d\n", i, blockResults[i], blockStarts[i], blockEnds[i]);
	}
	int globalMaxStretch = 0, currentStretch = blockResults[0];
	for (int i = 1; i < numBlocks; ++i) {
		if (blockEnds[i - 1] > 0 && blockStarts[i] > 0) {
			currentStretch += blockStarts[i] + blockEnds[i - 1] - 1;
		} else {
			globalMaxStretch = max(globalMaxStretch, currentStretch);
			currentStretch = blockResults[i];
		}
		globalMaxStretch = max(globalMaxStretch, blockResults[i]);
	}
	globalMaxStretch = max(globalMaxStretch, currentStretch);
	cudaFree(d_y);
	cudaFree(d_yBin);
	cudaFree(d_blockResults);
	cudaFree(d_blockStarts);
	cudaFree(d_blockEnds);
	free(blockResults);
	free(blockStarts);
	free(blockEnds);
	return (double)globalMaxStretch;
}

__global__ void countOneStretches(const int *yBin, int size, int *maxStretches, int *startStretches, int *endStretches) {
	extern __shared__ int s_data[];
	int tid = threadIdx.x;
	int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (globalIdx < size) {
		s_data[tid] = yBin[globalIdx];
	} else {
		s_data[tid] = 0;
	}
	__syncthreads();
	int localMaxStretch = 0;
	int count = (tid == 0 && globalIdx > 0) ? (yBin[globalIdx - 1] == 1) : 0;
	for (int i = 0; i < blockDim.x && globalIdx + i < size; i++) {
		if (s_data[i] == 1) {
			count++;
			localMaxStretch = max(localMaxStretch, count);
		} else {
			count = 0;
		}
	}
	if (tid == 0) {
		maxStretches[blockIdx.x] = localMaxStretch;
	}
	if (tid == 0) {
		int startStretch = 0;
		if (globalIdx > 0 && yBin[globalIdx - 1] == 1) {
			for (int i = 0; i < blockDim.x && i < size - globalIdx; i++) {
				if (s_data[i] == 1) startStretch++;
				else break;
			}
		}
		startStretches[blockIdx.x] = startStretch;
		int endStretch = 0;
		if (globalIdx + blockDim.x < size && yBin[globalIdx + blockDim.x] == 1) {
			for (int i = blockDim.x - 1; i >= 0 && globalIdx + i < size; i--) {
				if (s_data[i] == 1) endStretch++;
				else break;
			}
		}
		endStretches[blockIdx.x] = endStretch;
	}
}


__global__ void binarizeDiffs_1(const double *y, int *yBin, int size, double mean) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size - 1) {
		yBin[i] = (y[i] - mean <= 0) ? 0 : 1;
	}
}
double SB_BinaryStats_diff_longstretch1_CUDA(const double y[], int size) {	
	double *d_y;
	int *d_yBin, *d_blockResults, *d_blockStarts, *d_blockEnds;
	size_t bytes = size * sizeof(double);
	size_t bytesBin = (size - 1) * sizeof(int);
	double mean = compute_mean_cuda(y, size);
	cudaMalloc(&d_y, bytes);
	cudaMalloc(&d_yBin, bytesBin);
	int numBlocks = (size + 255) / 256;
	cudaMalloc(&d_blockResults, numBlocks * sizeof(int));
	cudaMalloc(&d_blockStarts, numBlocks * sizeof(int));
	cudaMalloc(&d_blockEnds, numBlocks * sizeof(int));
	cudaMemcpy(d_y, y, bytes, cudaMemcpyHostToDevice);
	binarizeDiffs_1<<<numBlocks, 256>>>(d_y, d_yBin, size, mean);
	countOneStretches<<<numBlocks, 256, 256 * sizeof(int)>>>(d_yBin, size - 1, d_blockResults, d_blockStarts, d_blockEnds);
	int *blockResults = (int *)malloc(numBlocks * sizeof(int));
	int *blockStarts = (int *)malloc(numBlocks * sizeof(int));
	int *blockEnds = (int *)malloc(numBlocks * sizeof(int));
	cudaMemcpy(blockResults, d_blockResults, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(blockStarts, d_blockStarts, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(blockEnds, d_blockEnds, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
	printf("Block Results:\n");
	for (int i = 0; i < numBlocks; ++i) {
		printf("Block %d: Max Stretch = %d, Start Stretch = %d, End Stretch = %d\n", i, blockResults[i], blockStarts[i], blockEnds[i]);
	}
	int globalMaxStretch = 0, currentStretch = blockResults[0];
	for (int i = 1; i < numBlocks; ++i) {
		if (blockEnds[i - 1] > 0 && blockStarts[i] > 0) {
			currentStretch += blockStarts[i] + blockEnds[i - 1] - 1;
		} else {
			globalMaxStretch = max(globalMaxStretch, currentStretch);
			currentStretch = blockResults[i];
		}
		globalMaxStretch = max(globalMaxStretch, blockResults[i]);
	}
	globalMaxStretch = max(globalMaxStretch, currentStretch);
	cudaFree(d_y);
	cudaFree(d_yBin);
	cudaFree(d_blockResults);
	cudaFree(d_blockStarts);
	cudaFree(d_blockEnds);
	free(blockResults);
	free(blockStarts);
	free(blockEnds);
	return (double)globalMaxStretch;
}


int main() {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	double *y = NULL;
	int size = 0;
	FILE *fp = fopen("/mnt/c/Users/yashw/Downloads/test.txt", "rb");
	if (!fp) {
		fprintf(stderr, "Failed to open the file.\n");
		return 1;
	}
	double temp;
	while (fscanf(fp, "%lf", &temp) == 1) {
		size++;
	}
	fseek(fp, 0, SEEK_SET);
	y = (double *)malloc(size * sizeof(double));
	if (!y) {
		fprintf(stderr, "Failed to allocate memory.\n");
		fclose(fp);
		return 1;
	}
	for (int i = 0; i < size; i++) {
		if (fscanf(fp, "%lf", &y[i]) != 1) {
			fprintf(stderr, "Failed to read data from file.\n");
			free(y);
			fclose(fp);
			return 1;
		}
	}
	fclose(fp);
	double ami = SB_BinaryStats_diff_longstretch0_CUDA(y, size);
	printf("SB_BinaryStats_diff_longstretch0_CUDA %f\n", ami);
	double ami_1 = SB_BinaryStats_diff_longstretch1_CUDA(y, size);
	printf("SB_BinaryStats_diff_longstretch1_CUDA %f\n", ami_1);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Elapsed time: %f ms\n", milliseconds);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	free(y);
	return 0;
}

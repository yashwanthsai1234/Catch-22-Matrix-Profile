#include <cuda_runtime.h>
#include<cuda_runtime.h>
#include<cufft.h>
#include<stdio.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <cmath>
__device__ double atomicAddDouble(double* address, double val) {
	    unsigned long long int* address_as_ull = (unsigned long long int*)address;
	        unsigned long long int old = *address_as_ull, assumed;

		    do {
			            assumed = old;
				            old = atomicCAS(address_as_ull, assumed,
							                            __double_as_longlong(val + __longlong_as_double(assumed)));
					        // Note: uses __double_as_longlong() and __longlong_as_double() to reinterpret
					        // the bits as double and back to 64-bit int for the atomicCAS operation.
					        } while (assumed != old);

		        return __longlong_as_double(old);
}



__global__ void findPotentialZeroCrossings(const double *autocorrs, int *zeroCrossingFlags, int size, int maxtau) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < maxtau && idx < size) {
		zeroCrossingFlags[idx] = (autocorrs[idx] <= 0) ? idx : maxtau;
	}
}

__global__ void reduceMinIndex(int *input, int size) {
	int idx = threadIdx.x;
	for (int offset = 1; offset < blockDim.x; offset *= 2) {
		if (idx % (2 * offset) == 0 && idx + offset < size) {
			if (input[idx + offset] < input[idx]) {
				input[idx] = input[idx + offset];
			}
		}
		__syncthreads();  // Ensure all threads have updated their values before next iteration
	}
}

__global__ void complexConjugateMultiply(cufftDoubleComplex *data, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		cufftDoubleComplex val = data[idx];
		data[idx] = cuCmul(val, make_cuDoubleComplex(val.x, -val.y)); // element * its conjugate
								        }
}

__global__ void normalizeAutocorr(double *data, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		data[idx] /= data[0]; // Normalize by the first element (maximum value)
	}
}

int nextpow2(int n){
	n--;
	n |= n >> 1;
	n |= n >> 2;
	n |= n >> 4;
	n |= n >> 8;
	n |= n >> 16;
	n++;
	return n;
}


double* cudaComputeAutocorrs(const double *y, int size) {
	int nFFT = nextpow2(size); // Assuming nextPow2 function is defined elsewhere
	// Allocate memory
	double *d_y, *autocorr;
	cufftDoubleComplex *d_freqDomain;
	cudaMalloc(&d_y, nFFT * sizeof(double));
	cudaMalloc(&d_freqDomain, nFFT * sizeof(cufftDoubleComplex));
	cudaMemset(d_y, 0, nFFT * sizeof(double)); // Zero padding
	cudaMemcpy(d_y, y, size * sizeof(double), cudaMemcpyHostToDevice);
	// Create CUFFT plans
	cufftHandle planForward, planInverse;
	cufftPlan1d(&planForward, nFFT, CUFFT_D2Z, 1);
	cufftPlan1d(&planInverse, nFFT, CUFFT_Z2D, 1);
	// Perform forward FFT
	cufftExecD2Z(planForward, d_y, d_freqDomain);
	// Compute complex conjugate multiplication
	int blockSize = 256;
	int numBlocks = (nFFT + blockSize - 1) / blockSize;
	complexConjugateMultiply<<<numBlocks, blockSize>>>(d_freqDomain, nFFT);
	// Perform inverse FFT
	cufftExecZ2D(planInverse, d_freqDomain, d_y);
	// Normalize the result
	normalizeAutocorr<<<numBlocks, blockSize>>>(d_y, nFFT);
	// Copy the result back to the host
	autocorr = (double*)malloc(nFFT * sizeof(double));
	cudaMemcpy(autocorr, d_y, nFFT * sizeof(double), cudaMemcpyDeviceToHost);
	// Cleanup
	cufftDestroy(planForward);
	cufftDestroy(planInverse);
	cudaFree(d_y);
	cudaFree(d_freqDomain);
	return autocorr;
}
int co_firstzero_cuda(const double y[], const int size, const int maxtau) {
	double *d_autocorrs;
	int *d_zeroCrossingFlags;
	int minIndex = maxtau;
	double *x = cudaComputeAutocorrs(y,size);  		        // Allocate memory on the device
	cudaMalloc((void **)&d_autocorrs, size * sizeof(double));
	cudaMalloc((void **)&d_zeroCrossingFlags, size * sizeof(int));
	// Copy data to the device
	cudaMemcpy(d_autocorrs, x, size * sizeof(double), cudaMemcpyHostToDevice);
	// Initialize zeroCrossingFlags with maxtau
	cudaMemset(d_zeroCrossingFlags, maxtau, size * sizeof(int));
	// Launch the findPotentialZeroCrossings kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	findPotentialZeroCrossings<<<blocksPerGrid, threadsPerBlock>>>(d_autocorrs, d_zeroCrossingFlags, size, maxtau);
	// Reduce to find the minimum index
	int numThreadsForReduce = 1024;  // Choose based on your device's capability
	reduceMinIndex<<<1, numThreadsForReduce>>>(d_zeroCrossingFlags, size);
	// Copy the result back to the host
	cudaMemcpy(&minIndex, d_zeroCrossingFlags, sizeof(int), cudaMemcpyDeviceToHost);
	// Free device memory
	cudaFree(d_autocorrs);
	cudaFree(d_zeroCrossingFlags);
	return minIndex;
}

__global__ void nanCheckKernel(const double *y, int size, int *nanFlag) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size && isnan(y[idx])) {
		atomicExch(nanFlag, 1);  // Set nanFlag to 1 if NaN is found
	}
}
__global__ void computeResKernel(const double *y, int size, int train_length, double *res) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size - train_length) {
		double yest = 0.0;
		for (int j = 0; j < train_length; j++) {
			yest += y[idx + j];
		}
		yest /= train_length;
		res[idx] = y[idx + train_length] - yest;
	}
}
__global__ void computeMeanKernel(const double *res, int size, double *mean) {
	extern __shared__ double sharedData[];
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;
	// Load data into shared memory
	sharedData[tid] = (idx < size) ? res[idx] : 0;
	__syncthreads();
	// Reduction to compute the sum
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sharedData[tid] += sharedData[tid + s];
		}
		__syncthreads();
	}
	// Compute the mean in the first thread of each block
	if (tid == 0) {
		 atomicAddDouble(mean, sharedData[0] / size);
	}
}

// Kernel to compute the variance of the residuals
__global__ void computeVarianceKernel(const double *res, int size, double mean, double *variance) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		double diff = res[idx] - mean;
		atomicAddDouble(variance, diff * diff / size);
	}
}

double FC_LocalSimple_mean_tauresrat_CUDA(const double y[], const int size, const int train_length) {
	double *d_y, *d_res;
	int *d_nanFlag;
	int nanFlag = 0;
	// Allocate memory on the device
	cudaMalloc((void **)&d_y, size * sizeof(double));
	cudaMalloc((void **)&d_res, (size - train_length) * sizeof(double));
	cudaMalloc((void **)&d_nanFlag, sizeof(int));
	// Copy data to the device
	cudaMemcpy(d_y, y, size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemset(d_nanFlag, 0, sizeof(int));  // Initialize nanFlag to 0
	// Launch the nanCheckKernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	nanCheckKernel<<<blocksPerGrid, threadsPerBlock>>>(d_y, size, d_nanFlag);
	// Copy the nanFlag back to host and check
	cudaMemcpy(&nanFlag, d_nanFlag, sizeof(int), cudaMemcpyDeviceToHost);
	if (nanFlag) {
		cudaFree(d_y);
		cudaFree(d_res);
		cudaFree(d_nanFlag);
		return NAN;  // Return if any NaN found
	}
	// Launch the computeResKernel
	computeResKernel<<<blocksPerGrid, threadsPerBlock>>>(d_y, size, train_length, d_res);
	// Copy res back to host
	double *res = (double *)malloc((size - train_length) * sizeof(double));
	cudaMemcpy(res, d_res, (size - train_length) * sizeof(double), cudaMemcpyDeviceToHost);
	// Continue with the sequential part...
	double resAC1stZ = co_firstzero_cuda(res, size - train_length, size - train_length);
	double yAC1stZ = co_firstzero_cuda(y, size, size);  // This can be optimized further if y is large
	double output = resAC1stZ / yAC1stZ;
	// This includes calls to co_firstzero and calculation of the final output.
	// Free device memory
	cudaFree(d_y);
	cudaFree(d_res);
	cudaFree(d_nanFlag);
	// Free host memory
	free(res);
	// Return the final output (placeholder, replace with actual computation)
	return output;
}
double FC_LocalSimple_mean_stderr_CUDA(const double y[], const int size, const int train_length) {
	double *d_y, *d_res, *d_mean, *d_variance;
	int *d_nanFlag;
	double mean, variance, stddev;
	int nanFlag = 0;
	// Allocate memory on the device
	cudaMalloc((void **)&d_y, size * sizeof(double));
	cudaMalloc((void **)&d_res, (size - train_length) * sizeof(double));
	cudaMalloc((void **)&d_mean, sizeof(double));
	cudaMalloc((void **)&d_variance, sizeof(double));
	cudaMalloc((void **)&d_nanFlag, sizeof(int));
	// Copy data to the device
	cudaMemcpy(d_y, y, size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemset(d_nanFlag, 0, sizeof(int));
	cudaMemset(d_mean, 0, sizeof(double));
	cudaMemset(d_variance, 0, sizeof(double));
	// Define kernel execution configuration
	int threadsPerBlock = 256;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	int sharedSize = threadsPerBlock * sizeof(double);
	// Launch kernels
	nanCheckKernel<<<blocksPerGrid, threadsPerBlock>>>(d_y, size, d_nanFlag);
	cudaMemcpy(&nanFlag, d_nanFlag, sizeof(int), cudaMemcpyDeviceToHost);
	if (nanFlag) {
		cudaFree(d_y);
		cudaFree(d_res);
		cudaFree(d_mean);
		cudaFree(d_variance);
		cudaFree(d_nanFlag);
		return NAN;  // Return NaN if any NaNs found in the input
	}
	computeResKernel<<<blocksPerGrid, threadsPerBlock>>>(d_y, size, train_length, d_res);
	// Compute the mean of residuals
	computeMeanKernel<<<blocksPerGrid, threadsPerBlock, sharedSize>>>(d_res, size - train_length, d_mean);
	// Compute the variance of residuals
  // Host variable for mean
	cudaMemcpy(&mean, d_mean, sizeof(double), cudaMemcpyDeviceToHost);
	computeVarianceKernel<<<blocksPerGrid, threadsPerBlock>>>(d_res, size - train_length, mean, d_variance);
	// Copy the variance back to host and compute standard deviation
	cudaMemcpy(&variance, d_variance, sizeof(double), cudaMemcpyDeviceToHost);
	stddev = sqrt(variance);
	// Clean up
	cudaFree(d_y);
	cudaFree(d_res);
	cudaFree(d_mean);
	cudaFree(d_variance);
	cudaFree(d_nanFlag);
	return stddev;
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
	// Count the number of values in the file
	double temp;
	// Count the number of values in the file
	while (fscanf(fp, "%lf", &temp) == 1) {
		size++;
	}
	// Reset the file position to the beginning
	fseek(fp, 0, SEEK_SET);
	// Allocate memory for the array
	y = (double *)malloc(size * sizeof(double));
	if (!y) {
		fprintf(stderr, "Failed to allocate memory.\n");
		fclose(fp);
		return 1;
	}
	// Read values into the array
	for (int i = 0; i < size; i++) {
		if (fscanf(fp, "%lf", &y[i]) != 1) {
			fprintf(stderr, "Failed to read data from file.\n");
			free(y);
			fclose(fp);
			return 1;
		}
	}

	fclose(fp);
	double result = FC_LocalSimple_mean_tauresrat_CUDA(y,size,1);
	printf("FC_LocalSimple_mean_tauresrat %f\n", result);
	double result_n = FC_LocalSimple_mean_stderr_CUDA(y,size,3);
	printf("FC_LocalSimple_mean_stderr_CUDA %f\n", result_n);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Elapsed time: %f ms\n", milliseconds);
	// Clean up
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	//cudaFree(autocorr_d);
	free(y);
	return 0;
}

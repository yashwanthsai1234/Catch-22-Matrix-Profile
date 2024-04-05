#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <cmath>



__global__ void mean_kernel(const double* d_input, double* d_partial_sums, int size) {
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < size) ? d_input[i] : 0;
    __syncthreads();
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) d_partial_sums[blockIdx.x] = sdata[0];
}

double compute_mean_cuda(const double* h_input, int size) {
    double *d_input, *d_partial_sums, *h_partial_sums;
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    int partialSumsSize = blocks * sizeof(double);

    cudaError_t cudaStatus = cudaMalloc(&d_input, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_input: %s\n", cudaGetErrorString(cudaStatus));
        return 0;
    }

    cudaStatus = cudaMalloc(&d_partial_sums, partialSumsSize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_partial_sums: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_input);
        return 0;
    }

    h_partial_sums = (double*)malloc(partialSumsSize);
    if (!h_partial_sums) {
        fprintf(stderr, "Failed to allocate host memory for h_partial_sums\n");
        cudaFree(d_input);
        cudaFree(d_partial_sums);
        return 0;
    }

    cudaStatus = cudaMemcpy(d_input, h_input, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_input);
        cudaFree(d_partial_sums);
        free(h_partial_sums);
        return 0;
    }

    mean_kernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(d_input, d_partial_sums, size);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "mean_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_input);
        cudaFree(d_partial_sums);
        free(h_partial_sums);
        return 0;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching mean_kernel: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
        cudaFree(d_input);
        cudaFree(d_partial_sums);
        free(h_partial_sums);
        return 0;
    }

    cudaStatus = cudaMemcpy(h_partial_sums, d_partial_sums, partialSumsSize, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_input);
        cudaFree(d_partial_sums);
        free(h_partial_sums);         
        return 0;
    }

    double totalSum = 0;
    for (int i = 0; i < blocks; i++) {
        totalSum += h_partial_sums[i];
    }

    cudaFree(d_input);
    cudaFree(d_partial_sums);
    free(h_partial_sums);
    return totalSum / size;
}

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include<float.h>

__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void histogramKernel(double *data, int size, double minVal, double maxVal, int nBins, double *histCounts, double binStep) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int binIdx = (int)((data[idx] - minVal) / binStep);
        if(binIdx >= 0 && binIdx < nBins) {
            atomicAddDouble(&(histCounts[binIdx]), 1.0);
        }
    }
}
int main() {
    int size = 270; // or read from file to set size dynamically
    double *h_data = (double*)malloc(size * sizeof(double));

    FILE *fp = fopen("/mnt/c/Users/yashw/Downloads/test.txt", "rb");
    if (!fp) {
        printf("Failed to open the file.\n");
        return 1;
    }
    //  fread(h_data, sizeof(double), size, fp);
    for(int i =0; i < size;i++){
        fscanf(fp, "%lf", &h_data[i]);
    }
    fclose(fp);

    double h_minVal = DBL_MAX, h_maxVal = -DBL_MAX;
    for (int i = 0; i < size; ++i) {
        if (h_data[i] < h_minVal) h_minVal = h_data[i];
            if (h_data[i] > h_maxVal) h_maxVal = h_data[i];
    }

    int nBins = 10; // Set No. of bins
    double binStep = (h_maxVal - h_minVal) / nBins;

    double *h_histCounts = (double*)malloc(nBins * sizeof(double));
    for (int i = 0; i < nBins; ++i) h_histCounts[i] = 0;

    double *d_data, *d_histCounts;
    cudaMalloc((void**)&d_data, size * sizeof(double));
    cudaMalloc((void**)&d_histCounts, nBins * sizeof(double));

    cudaMemcpy(d_data, h_data, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_histCounts, h_histCounts, nBins * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    histogramKernel<<<gridSize, blockSize>>>(d_data, size, h_minVal, h_maxVal, nBins, d_histCounts, binStep);

    cudaMemcpy(h_histCounts, d_histCounts, nBins * sizeof(double), cudaMemcpyDeviceToHost);
    double maxCount = 0;
    int numMaxs = 1;
    double mode = 0;
    for (int i = 0; i < nBins; ++i) {
        if (h_histCounts[i] > maxCount) {
            maxCount = h_histCounts[i];
            numMaxs = 1;
            mode = h_minVal + binStep * (i + 0.5);
        }else if (h_histCounts[i] == maxCount) {
            numMaxs += 1;
            mode += h_minVal + binStep * (i + 0.5);
        }
    }
    mode = mode / numMaxs;

    // Printing the mode
    printf("The mode is: %f\n", mode);
    
    cudaFree(d_data);
    cudaFree(d_histCounts);
    free(h_data);
    free(h_histCounts);

    return 0;
}

                                                                                                                                                                                                                                                                                                          94,0-1        Bot 
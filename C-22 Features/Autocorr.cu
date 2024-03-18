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

__global__ void subtract_mean_and_pad(double *d_y, int size, int nFFT, double mean) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_y[idx] -= mean;
    }
    else if (idx < nFFT) {
        d_y[idx] = 0.0;
    }
}

__global__ void complex_conjugate_multiply(cufftDoubleComplex *data, int nFFT) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nFFT) {
        cufftDoubleComplex val = data[idx];
        cufftDoubleComplex conjVal = cuConj(val);
        data[idx] = cuCmul(val, conjVal);
    }
}

__device__ double cufftComplex_abs(cufftDoubleComplex z) {
    return sqrt(z.x * z.x + z.y * z.y);
}

int nextpow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void normalize(double *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] /= data[0];
    }
}

double *cuda_co_autocorrs(const double *y, const int size) {
    int nFFT =  nextpow2(size);
    double *d_y, *autocorr;
    cufftDoubleComplex *d_freqDomain;
    cufftHandle plan_f, plan_i;
    cudaError_t cudaStatus;
    cufftResult cufftStatus;

    autocorr = (double *)malloc(nFFT * sizeof(double));
    if (!autocorr) {
        fprintf(stderr, "Failed to allocate host memory for autocorr\n");
        return NULL;
    }

    cudaStatus = cudaMalloc(&d_y, nFFT * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_y: %s\n", cudaGetErrorString(cudaStatus));
        free(autocorr);
        return NULL;
    }

    cudaStatus = cudaMalloc(&d_freqDomain, nFFT * sizeof(cufftDoubleComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_freqDomain: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_y);
        free(autocorr);
        return NULL;
    }

    cudaStatus = cudaMemcpy(d_y, y, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_y);
        cudaFree(d_freqDomain);
        free(autocorr);
        return NULL;
    }

    double *zeroPad = (double *)calloc(nFFT - size, sizeof(double));
    cudaMemcpy(d_y + size, zeroPad, (nFFT - size) * sizeof(double), cudaMemcpyHostToDevice);
    free(zeroPad);

    cufftStatus = cufftPlan1d(&plan_f, nFFT, CUFFT_D2Z, 1);
    if (cufftStatus != CUFFT_SUCCESS) {
        fprintf(stderr, "cufftPlan1d failed with error code %d\n", cufftStatus);
        cudaFree(d_y);
        cudaFree(d_freqDomain);
        free(autocorr);
        return NULL;
    }

    cufftStatus = cufftPlan1d(&plan_i, nFFT, CUFFT_Z2D, 1);
    if (cufftStatus != CUFFT_SUCCESS) {
        fprintf(stderr, "cufftPlan1d failed with error code %d\n", cufftStatus);
        cudaFree(d_y);
        cudaFree(d_freqDomain);
        free(autocorr);
        return NULL;
    }

    cufftStatus = cufftExecD2Z(plan_f, d_y, d_freqDomain);
    if (cufftStatus != CUFFT_SUCCESS) {
        fprintf(stderr, "cufftExecD2Z failed with error code %d\n", cufftStatus);
        cudaFree(d_y);
        cudaFree(d_freqDomain);
        free(autocorr);
        return NULL;
    }

    int threadsPerBlock = 256;
    int blocks = (nFFT + threadsPerBlock - 1) / threadsPerBlock;
    complex_conjugate_multiply<<<blocks, threadsPerBlock>>>(d_freqDomain, nFFT);
    cudaDeviceSynchronize();

    cufftStatus = cufftExecZ2D(plan_i, d_freqDomain, d_y);
    if (cufftStatus != CUFFT_SUCCESS) {
        fprintf(stderr, "cufftExecZ2D failed with error code %d\n", cufftStatus);
        cufftDestroy(plan_i);
        free(autocorr);
        return NULL;
    }

    int blocksPerGrid = (nFFT + threadsPerBlock - 1) / threadsPerBlock;
    normalize<<<blocksPerGrid, threadsPerBlock>>>(d_y, nFFT);
    cudaDeviceSynchronize();

    cudaStatus = cudaMemcpy(autocorr, d_y, nFFT * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_y);
        cudaFree(d_freqDomain);
        free(autocorr);
        return NULL;
    }

    cufftDestroy(plan_f);
    cufftDestroy(plan_i);
    cudaFree(d_y);
    cudaFree(d_freqDomain);
    return autocorr;
}

__global__ void find_first_min_kernel(const double *autocorrs, int size, int *minIndex) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i < size - 1 && autocorrs[i] < autocorrs[i - 1] && autocorrs[i] < autocorrs[i + 1]) {
        atomicMin(minIndex, i);
    }
}

int CO_FirstMin_ac_cuda(const double y[], const int size) {
    double *autocorrs = cuda_co_autocorrs(y, size);
    double *d_autocorrs;
    int *d_minIndex;
    int h_minIndex = size;
    cudaMalloc(&d_autocorrs, size * sizeof(double));
    cudaMalloc(&d_minIndex, sizeof(int));
    cudaMemcpy(d_autocorrs, autocorrs, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_minIndex, &h_minIndex, sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    find_first_min_kernel<<<blocks, threadsPerBlock>>>(d_autocorrs, size, d_minIndex);

    cudaMemcpy(&h_minIndex, d_minIndex, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_autocorrs);
    cudaFree(d_minIndex);
    free(autocorrs);
    return h_minIndex;
}

__global__ void findThresholdCrossing(const double* autocorr, int size, double thresh, int* crossingIndex) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0 && idx < size) {
        if (autocorr[idx] < thresh && autocorr[idx - 1] >= thresh) {
            atomicMin(crossingIndex, idx);
        }
    }
}

double CO_f1ecac_CUDA(const double* y, int size) {
    double* autocorr_d = nullptr;
    int* crossingIndex_d = nullptr;
    int crossingIndex_h = INT_MAX;
    cudaMalloc((void**)&autocorr_d, size * sizeof(double));
    cudaMemcpy(autocorr_d, y, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&crossingIndex_d, sizeof(int));
    cudaMemcpy(crossingIndex_d, &crossingIndex_h, sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    double thresh = 1.0 / exp(1);
    findThresholdCrossing<<<blocksPerGrid, threadsPerBlock>>>(autocorr_d, size, thresh, crossingIndex_d);

    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching findThresholdCrossing!\n", cudaStatus);
    }

    cudaMemcpy(&crossingIndex_h, crossingIndex_d, sizeof(int), cudaMemcpyDeviceToHost);
    double out = (double)size;
    if (crossingIndex_h != INT_MAX && crossingIndex_h > 0 && crossingIndex_h < size) {
        double autocorr_values[2];
        cudaMemcpy(autocorr_values, &autocorr_d[crossingIndex_h - 1], 2 * sizeof(double), cudaMemcpyDeviceToHost);
        double m = autocorr_values[1] - autocorr_values[0];
        double dy = thresh - autocorr_values[0];
        double dx = dy / m;
        out = crossingIndex_h - 1 + dx;
    } else {
        printf("Threshold crossing not found.\n");
    }

    cudaFree(autocorr_d);
    cudaFree(crossingIndex_d);
    return out;
}

__global__ void compute_cubed_differences(const double *y, double *cubed_diffs, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size - 1) {
        double diff = y[idx + 1] - y[idx];
        cubed_diffs[idx] = diff * diff * diff;
    }
}

double CO_trev_1_num_cuda(const double *y, int size) {
    double *d_y, *d_cubed_diffs;
    cudaMalloc(&d_y, size * sizeof(double));
    cudaMemcpy(d_y, y, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&d_cubed_diffs, (size - 1) * sizeof(double));
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    compute_cubed_differences<<<blocks, threadsPerBlock>>>(d_y, d_cubed_diffs, size);
    double mean_cubed_diffs = compute_mean_cuda(d_cubed_diffs, size - 1);
    cudaFree(d_y);
    cudaFree(d_cubed_diffs);
    return mean_cubed_diffs;
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
    int firstMinIndex = CO_FirstMin_ac_cuda(y, size);
    printf("CO_First_min: %d\n", firstMinIndex);

    double *autocorr_d = cuda_co_autocorrs(y, size);
    float result = CO_f1ecac_CUDA(autocorr_d, size);
    float result_2 = CO_trev_1_num_cuda(y, size);
    printf("CO_F1ecac : %f\n", result);
    printf("CO_trev_num1 : %.14f\n", result_2);

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

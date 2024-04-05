#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

__global__ void prepareWindowedSignal(const double *y, const double *window, cufftDoubleComplex *d_F, int windowWidth, int NFFT, int offset, double m) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < windowWidth) {
        double windowedValue = (y[idx + offset] - m) * window[idx];
        d_F[idx].x = windowedValue;  
        d_F[idx].y = 0.0;            
    } else if (idx < NFFT) {
        d_F[idx].x = 0.0;  
        d_F[idx].y = 0.0;  
    }
}

void computeFFT(cufftHandle plan, cufftDoubleComplex *d_F, int NFFT) {
    cufftExecZ2Z(plan, d_F, d_F, CUFFT_FORWARD);
}

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
    int blocks = (270+size + threadsPerBlock - 1) / threadsPerBlock;
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

int welchCuda(const double y[], const int size, const int NFFT, const double Fs, const double window[], const int windowWidth, double **d_Pxx, double **d_f) {
    double dt = 1.0 / Fs;
    double df = Fs / NFFT;
    int k = floor((double)size / ((double)windowWidth / 2.0)) - 1;
    double *d_y, *d_window;
    double m = compute_mean_cuda(y, size);
    cufftDoubleComplex *d_F;
    cudaMalloc(&d_y, size * sizeof(double));
    cudaMemcpy(d_y, y, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&d_window, windowWidth * sizeof(double));
    cudaMemcpy(d_window, window, windowWidth * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&d_F, NFFT * sizeof(cufftDoubleComplex));
    cufftHandle plan;
    cufftPlan1d(&plan, NFFT, CUFFT_Z2Z, 1);
    double *P = (double *)malloc(NFFT * sizeof(double));
    for (int i = 0; i < NFFT; i++) {
        P[i] = 0;
    }
    for (int i = 0; i < k; i++) {
        int offset = i * windowWidth / 2;
        int threadsPerBlock = 256;
        int blocksPerGrid = (NFFT + threadsPerBlock - 1) / threadsPerBlock;
        prepareWindowedSignal<<<blocksPerGrid, threadsPerBlock>>>(d_y, d_window, d_F, windowWidth, NFFT, offset, m);
        computeFFT(plan, d_F, NFFT);
        cufftDoubleComplex *F = (cufftDoubleComplex *)malloc(NFFT * sizeof(cufftDoubleComplex));
        cudaMemcpy(F, d_F, NFFT * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
        for (int l = 0; l < NFFT; l++) {
            P[l] += (F[l].x * F[l].x + F[l].y * F[l].y);
        }    
        free(F);
    }
    cufftDestroy(plan);
    int Nout = (NFFT / 2 + 1);
    cudaMalloc((void**)d_Pxx, Nout * sizeof(double));
    cudaMalloc((void**)d_f, Nout * sizeof(double));
    double* temp_f = (double*)malloc(Nout * sizeof(double));
    double* temp_Pxx = (double*)malloc(Nout * sizeof(double));
    for (int i = 0; i < Nout; i++) {
        temp_f[i] = i * df;
        temp_Pxx[i] = P[i] / (k * windowWidth) * dt;
        if (i > 0 && i < Nout - 1) temp_Pxx[i] *= 2;
    }
    cudaMemcpy(*d_Pxx, temp_Pxx, Nout * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_f, temp_f, Nout * sizeof(double), cudaMemcpyHostToDevice);
    free(P);
    free(temp_f);
    free(temp_Pxx);
    cudaFree(d_y);
    cudaFree(d_window);
    cudaFree(d_F);
    return Nout;
}

__global__ void calculateAngularFrequencyAndSpectrum(const double* f, double* w, double* Sw, int nWelch, double PI) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < nWelch) {
        w[idx] = 2 * PI * f[idx];
        Sw[idx] = Sw[idx] / (2 * PI);
    }
}

void cumsum(const double a[], const int size, double b[]) {
    b[0] = a[0];
    for (int i = 1; i < size; i++) {
        b[i] = a[i] + b[i-1];
    }   
}

double SP_Summaries_welch_rect_cuda(const double y[], int size, const char what[]) {
    for (int i = 0; i < size; ++i) {
        if (isnan(y[i])) return NAN;
    }
    double Fs = 1.0;
    int NFFT = pow(2, ceil(log2(size)));
    int windowWidth = size;
    double* window = (double*)malloc(windowWidth * sizeof(double));
    for (int i = 0; i < windowWidth; ++i) {
        window[i] = 1.0;
    }
    double* d_Pxx = nullptr;
    double* d_f = nullptr;
    double* d_window;
    cudaMalloc(&d_window, windowWidth * sizeof(double));
    cudaMemcpy(d_window, window, windowWidth * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_Pxx, NFFT / 2 + 1 * sizeof(double));
    cudaMalloc((void**)&d_f, NFFT / 2 + 1 * sizeof(double));  
    int nWelch = welchCuda(y, size, NFFT, Fs, d_window, windowWidth, &d_Pxx, &d_f);
    free(window);
    cudaFree(d_window);
    double* Pxx = (double*)malloc(nWelch * sizeof(double));
    double* f = (double*)malloc(nWelch * sizeof(double));
    cudaMemcpy(Pxx, d_Pxx, nWelch * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(f, d_f, nWelch * sizeof(double), cudaMemcpyDeviceToHost);
    double PI = 3.14159265359;
    double* w = (double*)malloc(nWelch * sizeof(double));
    double* Sw = (double*)malloc(nWelch * sizeof(double));
    for (int i = 0; i < nWelch; i++) {
        w[i] = 2 * PI * f[i];
        Sw[i] = Pxx[i] / (2 * PI);
    }
    double dw = w[1] - w[0];
    double* csS = (double*)malloc(nWelch * sizeof(double));
    cumsum(Sw, nWelch, csS);
    double output = 0.0;
    if (strcmp(what, "centroid") == 0) {
        double csSThres = csS[nWelch - 1] * 0.5;
        double centroid = 0.0;
        for (int i = 0; i < nWelch; i++) {
            if (csS[i] > csSThres) {
                centroid = w[i];
                break;
            }
        }
        output = centroid;
        free(csS); 
    } else if (strcmp(what, "area_5_1") == 0) {
        int limit = nWelch / 5;
        double area_5_1 = 0.0;
        for (int i = 0; i < limit; i++) {
            area_5_1 += Sw[i];
        }
        area_5_1 *= dw;
        output = area_5_1;
    }
    cudaFree(d_Pxx);
    cudaFree(d_f);
    free(Pxx);
    free(f);
    free(w);
    free(Sw);
    return output;
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
    double area_5_1 = SP_Summaries_welch_rect_cuda(y, size, "area_5_1");
    printf("Area under the first 1/5th of the spectrum: %f\n", area_5_1);
    double centroid = SP_Summaries_welch_rect_cuda(y, size, "centroid");
    printf("Spectral Centroid: %f\n", centroid);
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

#include <stdio.h>
#include <math.h>

double mean(const double *a, const int size);

__global__ void autocorr_lag_kernel(const double *x, double *result, const int size, const int lag) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size - lag) {
        result[idx] = (x[idx] - x[idx + lag]) / (double)size;
    }
}

__global__ void corr_kernel(const double *x, const double *y, double *nom, double *denomX, double *denomY, const int size, const double meanX, const double meanY) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        nom[idx] = (x[idx] - meanX) * (y[idx] - meanY);
        denomX[idx] = (x[idx] - meanX) * (x[idx] - meanX);
        denomY[idx] = (y[idx] - meanY) * (y[idx] - meanY);
    }
}

__global__ void mean_kernel(const double *a, double *result, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        *result += a[idx];
    }
}

double IN_AutoMutualInfoStats_40_gaussian_fmmi(const double *y, const int size) {
    // NaN check
    for (int i = 0; i < size; i++) {
        if (isnan(y[i])) {
            return NAN;
        }
    }

    // maximum time delay
    int tau = 40;

    // don't go above half the signal length
    if (tau > ceil((double)size / 2)) {
        tau = ceil((double)size / 2);
    }

    // Allocate device memory
    double *d_y, *d_ami;
    cudaMalloc((void**)&d_y, size * sizeof(double));
    cudaMalloc((void**)&d_ami, size * sizeof(double));

    // Copy input data to device
    cudaMemcpy(d_y, y, size * sizeof(double), cudaMemcpyHostToDevice);

    // Launch autocorrelations kernel
    autocorr_lag_kernel<<<(size - tau + 255) / 256, 256>>>(d_y, d_ami, size, tau);

    // Copy results back to host
    double *ami = (double*)malloc(size * sizeof(double));
    cudaMemcpy(ami, d_ami, size * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_y);
    cudaFree(d_ami);

    // compute automutual information
    double fmmi = tau;
    for (int i = 1; i < tau - 1; i++) {
        if (ami[i] < ami[i - 1] && ami[i] < ami[i + 1]) {
            fmmi = i;
            break;
        }
    }

    free(ami);

    return fmmi;
}

double autocorr_lag(const double *x, const int size, const int lag) {
    double *d_result;
    cudaMalloc((void**)&d_result, (size - lag) * sizeof(double));

    autocorr_lag_kernel<<<(size - lag + 255) / 256, 256>>>(x, d_result, size, lag);

    double *result = (double*)malloc((size - lag) * sizeof(double));
    cudaMemcpy(result, d_result, (size - lag) * sizeof(double), cudaMemcpyDeviceToHost);

    double ac = 0.0;
    for (int i = 0; i < size - lag; i++) {
        ac += result[i];
    }

    free(result);
    cudaFree(d_result);

    return ac;
}

double corr(const double *x, const double *y, const int size) {
    double *d_nom, *d_denomX, *d_denomY;
    cudaMalloc((void**)&d_nom, size * sizeof(double));
    cudaMalloc((void**)&d_denomX, size * sizeof(double));
    cudaMalloc((void**)&d_denomY, size * sizeof(double));

    double meanX = mean(x, size);
    double meanY = mean(y, size);

    corr_kernel<<<(size + 255) / 256, 256>>>(x, y, d_nom, d_denomX, d_denomY, size, meanX, meanY);

    double *nom = (double*)malloc(size * sizeof(double));
    double *denomX = (double*)malloc(size * sizeof(double));
    double *denomY = (double*)malloc(size * sizeof(double));

    cudaMemcpy(nom, d_nom, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(denomX, d_denomX, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(denomY, d_denomY, size * sizeof(double), cudaMemcpyDeviceToHost);

    double nom_sum = 0.0;
    double denomX_sum = 0.0;
    double denomY_sum = 0.0;

    for (int i = 0; i < size; i++) {
        nom_sum += nom[i];
        denomX_sum += denomX[i];
        denomY_sum += denomY[i];
    }

    free(nom);
    free(denomX);
    free(denomY);
    cudaFree(d_nom);
    cudaFree(d_denomX);
    cudaFree(d_denomY);

    return nom_sum / sqrt(denomX_sum * denomY_sum);
}

double mean(const double *a, const int size) {
    double *d_result;
    cudaMalloc((void**)&d_result, sizeof(double));

    mean_kernel<<<(size + 255) / 256, 256>>>(a, d_result, size);

    double *result = (double*)malloc(sizeof(double));
    cudaMemcpy(result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    free(result);
    cudaFree(d_result);

    return *result / size;
}

int main() {
    // Example usage with some dummy data
    const int size = 1000;
    double *data = (double*)malloc(size * sizeof(double));
    FILE *fp = fopen("/mnt/c/Users/yashw/Downloads/test.txt", "rb");
    if (!fp) {
        printf("Failed to open the file.\n");
        return 1;
    }

    for (int i = 0; i < size; i++) {
        fscanf(fp, "%lf", &data[i]);
    }

    fclose(fp);

    // Populate data with some values
    double result = IN_AutoMutualInfoStats_40_gaussian_fmmi(data, size);

    printf("Result: %f\n", result);

    // Free allocated memory
    free(data);

    return 0;
}

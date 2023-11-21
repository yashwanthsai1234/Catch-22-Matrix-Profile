#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void computeCubedDifferences(const double *y, int size, double *diffCubed)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size-1) {
                double diff = y[idx + 1] - y[idx];
                diffCubed[idx] = pow(diff, 3);
        }
}
double CO_trev_1_num_CUDA(const double y[], const int size)
{           // NaN check
        for (int i = 0; i < size; i++) {
                if (isnan(y[i])) {
                        return NAN;
                }
        }

        double *d_y, *d_diffCubed;
        cudaMalloc((void**)&d_y, size * sizeof(double));
        cudaMemcpy(d_y, y, size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_diffCubed, (size - 1) * sizeof(double));
        int threadsPerBlock = 256;
        int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
        // Launch kernel to compute cubed differences
        computeCubedDifferences<<<blocks, threadsPerBlock>>>(d_y, size, d_diffCubed);

        // Check for errors
        cudaError_t cudaErr = cudaGetLastError();
        if (cudaErr != cudaSuccess) {
                fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaErr));
                cudaFree(d_y);
                cudaFree(d_diffCubed);
                return NAN;
        }
cudaDeviceSynchronize();
        // Copy results back to host
        double *diffCubed = (double *)malloc((size - 1) * sizeof(double));
        cudaMemcpy(diffCubed, d_diffCubed, (size - 1) * sizeof(double), cudaMemcpyDeviceToHost);
        for(int i = 0; i < 277 && i < size - 1; i++) {
                printf("The cubed value at index %d is %.20f\n", i, diffCubed[i]);
        }
        // Compute the mean of cubed differences
        double sum = 0;
        for (int i = 0; i < size - 1; i++) {
                sum += diffCubed[i];
        }
        double mean = sum / (size - 1);
        // Cleanup
        free(diffCubed);
        cudaFree(d_y);
        cudaFree(d_diffCubed);
        return mean;
}

int main()
{
        const int size = 270; // Replace with your actual size
        double *y = (double *)malloc(size * sizeof(double));
        FILE *fp = fopen("/mnt/c/Users/yashw/Downloads/test.txt", "rb");
        if (!fp) {
                fprintf(stderr, "Failed to open the file.\n");
                return 1;
        }
        for (int i = 0; i < size; i++) {
                fscanf(fp, "%lf", &y[i]);
        }
        //for(int i = 0; i < 20; i++)
        //      printf("The top Values are = %f\n", y[i]);
        fclose(fp);
// Example usage of CO_trev_1_num_CUDA
        double result = CO_trev_1_num_CUDA(y, size);
        printf("CO_trev_1_num_CUDA result: %.20f\n", result);
        free(y);
        return 0;
}

                                                            
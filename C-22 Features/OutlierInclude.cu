#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/count.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <cmath>
#include <algorithm>

__global__ void markOutliers(const double *data, int *outlierFlags, double threshold, int size, int sign) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        outlierFlags[idx] = sign > 0 ? data[idx] >= threshold : data[idx] <= -threshold;
    }
}

__global__ void calculateDistances(const int *indices, double *distances, int numIndices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numIndices - 1) {
        distances[idx] = indices[idx + 1] - indices[idx];
    }
}

double DN_OutlierInclude_np_001_mdrmd_CUDA(const double *y_host, int size, int sign) {
    double *data_dev = nullptr;
    cudaMalloc(&data_dev, size * sizeof(double));
    cudaMemcpy(data_dev, y_host, size * sizeof(double), cudaMemcpyHostToDevice);

    int *outlierFlags_dev = nullptr;
    cudaMalloc(&outlierFlags_dev, size * sizeof(int));

    double maxValue = 0;
    if (sign > 0) {
        thrust::device_ptr<double> dev_ptr(data_dev);
        maxValue = *thrust::max_element(dev_ptr, dev_ptr + size);
    } else {
        thrust::device_ptr<double> dev_ptr(data_dev);
        double minValue = *thrust::min_element(dev_ptr, dev_ptr + size);
        maxValue = fabs(minValue);
    }

    double inc = 0.01;
    int nThresh = static_cast<int>(maxValue / inc) + 1;

    thrust::device_vector<double> msDti1(nThresh), msDti4(nThresh);
    thrust::host_vector<double> msDti3(nThresh);

    int blockSize = 256;
    for (int j = 0; j < nThresh; ++j) {
        double threshold = j * inc;

        markOutliers<<<(size + blockSize - 1) / blockSize, blockSize>>>(data_dev, outlierFlags_dev, threshold, size, sign);
        cudaDeviceSynchronize();

        thrust::device_vector<int> indices(size);
        thrust::sequence(indices.begin(), indices.end());

        thrust::device_vector<int> outlierIndices(size);
        auto end = thrust::copy_if(thrust::device, indices.begin(), indices.end(), outlierFlags_dev, outlierIndices.begin(), [] __device__(int flag) { return flag == 1; });
        int numOutliers = end - outlierIndices.begin();

        if (numOutliers > 1) {
            thrust::device_vector<double> distances(numOutliers - 1);
            calculateDistances<<<(numOutliers - 1 + blockSize - 1) / blockSize, blockSize>>>(thrust::raw_pointer_cast(outlierIndices.data()), thrust::raw_pointer_cast(distances.data()), numOutliers);
            cudaDeviceSynchronize();

            msDti1[j] = thrust::reduce(distances.begin(), distances.end()) / (numOutliers - 1);
            msDti4[j] = (outlierIndices[numOutliers / 2] - size / 2.0) / (size / 2.0);
        } else {
            msDti1[j] = 0;
            msDti4[j] = 0;
        }

        msDti3[j] = static_cast<double>(numOutliers) / size * 100.0;
    }

    thrust::sort(msDti4.begin(), msDti4.end());
    double outputScalar = msDti4[nThresh / 2];

    cudaFree(data_dev);
    cudaFree(outlierFlags_dev);

    return outputScalar;
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
    double sign = 1.0;
    double result = DN_OutlierInclude_np_001_mdrmd_CUDA(y, size, sign);
    printf("The OutlierInclude_np_001_mdrmd is %f\n", result);

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

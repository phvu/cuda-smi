#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>

#define CUDA_CALL(function, ...)  { \
    cudaError_t status = function(__VA_ARGS__); \
    anyCheck(status == cudaSuccess, cudaGetErrorString(status), #function, __FILE__, __LINE__); \
}

void anyCheck(bool is_ok, const char *description, const char *function, const char *file, int line) {
    if (!is_ok) {
        fprintf(stderr,"Error: %s in %s at %s:%d\n", description, function, file, line);
        exit(EXIT_FAILURE);
    }
}

int main() {
    int cudaDeviceCount;
    struct cudaDeviceProp deviceProp;
    size_t memFree, memTotal;

    CUDA_CALL(cudaGetDeviceCount, &cudaDeviceCount);

    for (int deviceId = 0; deviceId < cudaDeviceCount; ++deviceId) {
        CUDA_CALL(cudaGetDeviceProperties, &deviceProp, deviceId);
        
        printf("Device %2d", deviceId);
        printf(" [PCIe %04x:%02x:%02x.0]", deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);
        printf(": %20s (CC %d.%d)", deviceProp.name, deviceProp.major, deviceProp.minor);
        CUDA_CALL(cudaMemGetInfo, &memFree, &memTotal);
        printf(": %5zu of %5zu MiB Free", memFree, memTotal);
        printf("\n");
    }
    return 0;
}


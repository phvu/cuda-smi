#include <iostream>
#include <iomanip>
#include <cuda_runtime_api.h>

#define CUDA_CALL(function, ...)  { \
    cudaError_t status = function(__VA_ARGS__); \
    anyCheck(status == cudaSuccess, cudaGetErrorString(status), #function, __FILE__, __LINE__); \
}

void anyCheck(bool is_ok, const char *description, const char *function, const char *file, int line) {
    if (!is_ok) {
        std::cout << "Error: " << description << " in " << function << " at " << file << ":" << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int cudaDeviceCount;
    struct cudaDeviceProp deviceProp;
    size_t memFree, memTotal;

    CUDA_CALL(cudaGetDeviceCount, &cudaDeviceCount);

    for (int deviceId = 0; deviceId < cudaDeviceCount; ++deviceId) {
        CUDA_CALL(cudaSetDevice, deviceId);
        CUDA_CALL(cudaGetDeviceProperties, &deviceProp, deviceId);

        //std::cout.imbue(std::locale("en_US.utf8"));
        std::cout << "Device " << deviceId;
        std::cout << " [PCIe " << deviceProp.pciDomainID << ":" << deviceProp.pciBusID
                  << ":" << deviceProp.pciDeviceID << ".0]";
        std::cout << ": " << deviceProp.name << " (CC " << deviceProp.major << "." << deviceProp.minor << ")";
        CUDA_CALL(cudaMemGetInfo, &memFree, &memTotal);
        std::cout << ": " << std::setprecision(5) << memFree/(1024*1024.) 
                  << " of " << memTotal/(1024*1024.) << " MB (i.e. "
                  << std::setprecision(3) << 100*memFree/(float)memTotal << "%) Free"
                  << std::endl;
    }
    return 0;
}


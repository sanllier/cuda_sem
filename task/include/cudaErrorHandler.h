#ifndef ERROR_HANDLER_H
#define ERROR_HANDLER_H

#include <cstdio>
#include <cstdlib>

//---------------------------------------------------------------

#define SAFE_CALL( CallInstruction ) \
{ \
    cudaError_t cuerr = CallInstruction; \
    if(cuerr != cudaSuccess) \
    { \
        printf("CUDA error: %s at call \"" #CallInstruction "\"\n", cudaGetErrorString(cuerr)); \
        exit(0); \
    } \
}

#define SAFE_KERNEL_CALL( KernelCallInstruction ) \
{ \
    KernelCallInstruction; \
    cudaError_t cuerr = cudaGetLastError(); \
    if(cuerr != cudaSuccess) \
    { \
        printf("CUDA error in kernel launch: %s at kernel \"" #KernelCallInstruction "\"\n", cudaGetErrorString(cuerr)); \
        exit(0); \
    } \
    cuerr = cudaDeviceSynchronize(); \
    if(cuerr != cudaSuccess) \
    { \
        printf("CUDA error in kernel execution: %s at kernel \"" #KernelCallInstruction "\"\n", cudaGetErrorString(cuerr)); \
        exit(0); \
    } \
}

//---------------------------------------------------------------

#endif

#include "defs.h"
#include "cudaErrorHandler.h"
#include "cuda_runtime.h"

#include <cstdio>

//---------------------------------------------------------------

__global__ void partOneTwoKernel( cudaPitchedPtr aDev, cudaPitchedPtr bDev, cudaPitchedPtr cDev, int aH, int aW, int bW )
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if( row > aH || col > bW ) 
        return;

    MATRIX_TYPE temp = MATRIX_TYPE(0);
    for ( int i = 0; i < aW; ++i )
        temp += get_elem( aDev, row, i ) * get_elem( bDev, i, col );

    get_elem( cDev, row, col ) = temp;
}

//---------------------------------------------------------------

__global__ void partThreeKernel( cudaPitchedPtr aDev, cudaPitchedPtr bDev, cudaPitchedPtr cDev, int aH, int aW, int bW )
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int n = aDev.xsize / blockDim.x;

    printf( "%d\r\n", n );

    extern __shared__ MATRIX_TYPE aShared[];
    MATRIX_TYPE* bShared = aShared + blockDim.x * blockDim.y;

    for ( int i = 0; i < n; ++i )
    {
       aShared[ threadIdx.y * blockDim.x + threadIdx.x ] = 0;//get_elem( aDev, row + threadIdx.y, i * blockDim.x + threadIdx.x );
       bShared[ threadIdx.y * blockDim.x + threadIdx.x ] = 0;//get_elem( bDev, i * blockDim.y + threadIdx.y, col + threadIdx.x );

        //__syncthreads();
    }

    get_elem( cDev, row, col ) = 0;
}

//---------------------------------------------------------------
//---------------------------------------------------------------
//---------------------------------------------------------------

void launchPartOneKernel( cudaPitchedPtr aDev, cudaPitchedPtr bDev, cudaPitchedPtr cDev, int aH, int aW, int bW, int blockH, int blockW )
{
   dim3 block = dim3 ( blockH, blockW );
   dim3 grid = dim3 ( aH / blockH + 1, bW / blockW + 1 );
   SAFE_KERNEL_CALL( ( partOneTwoKernel<<< grid, block >>>( aDev, bDev, cDev, aH, aW, bW ) ) );
}

//---------------------------------------------------------------

void launchPartTwoKernel( cudaPitchedPtr aDev, cudaPitchedPtr bDev, cudaPitchedPtr cDev, int aH, int aW, int bW, int blockH, int blockW )
{
   dim3 block = dim3 ( blockH, blockW );
   dim3 grid = dim3 ( aH / blockH + 1, bW / blockW + 1 );
   SAFE_KERNEL_CALL( ( partOneTwoKernel<<< grid, block >>>( aDev, bDev, cDev, aH, aW, bW ) ) );
}

//---------------------------------------------------------------

void launchPartThreeKernel( cudaPitchedPtr aDev, cudaPitchedPtr bDev, cudaPitchedPtr cDev, int aH, int aW, int bW, int blockH, int blockW )
{
   dim3 block = dim3 ( blockH, blockW );
   dim3 grid = dim3 ( aH / blockH, bW / blockW );
   int shared = blockH * blockW * sizeof( MATRIX_TYPE ) * 2;
   SAFE_KERNEL_CALL( ( partThreeKernel<<< grid, block, shared >>>( aDev, bDev, cDev, aH, aW, bW ) ) );
}

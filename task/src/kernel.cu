#include "defs.h"
#include "cudaErrorHandler.h"
#include "cuda_runtime.h"

#define get_elem( arr, Row, Column ) ( ((Type*)( (char*)arr.ptr + (Row)*arr.pitch ) )[ Column ])

//---------------------------------------------------------------

__global__ void sum_kernel( cudaPitchedPtr aDev, cudaPitchedPtr bDev, cudaPitchedPtr cDev, int aH, int aW, int bW )
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if( row > aH || col > bW ) 
        return;

    MATRIX_TYPE temp = MATRIX_TYPE(0);
    for ( int i = 0; i < aW; ++i )
        temp += get_elem( aDev, row, i ) * get_elem( bDev, i, col );        get_elem( cDev, bW, col ) = temp;
}

//---------------------------------------------------------------

void launchKernel( cudaPitchedPtr aDev, cudaPitchedPtr bDev, cudaPitchedPtr cDev, int aH, int aW, int bW )
{
   dim3 threadsInBlock = dim3 ( BLOCK_SIZE );
   dim3 blocksInGrid = dim3 ( aH / BLOCK_SIZE + bW / BLOCK_SIZE );
   SAFE_KERNEL_CALL( ( sum_kernel<<< blocksInGrid,threadsInBlock >>>( aDev, bDev, cDev, n ) ) );
}

//---------------------------------------------------------------

#include "defs.h"
#include "cudaErrorHandler.h"
#include "cuda_runtime.h"

#include <cstdio>

#define get_elem( _pptr_, _row_, _col_ ) ( *( ( MATRIX_TYPE* )( ( char* )_pptr_.ptr + _row_ * _pptr_.pitch ) + _col_ ) )

//---------------------------------------------------------------

__global__ void sum_kernel( cudaPitchedPtr aDev, cudaPitchedPtr bDev, cudaPitchedPtr cDev, int aH, int aW, int bW )
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if( row > aH || col > bW || row < 0 || col < 0 ) 
        return;

    MATRIX_TYPE temp = MATRIX_TYPE(0);
    for ( int i = 0; i < aW; ++i )
        temp += get_elem( aDev, row, i ) * get_elem( bDev, i, col );

    get_elem( cDev, row, col ) = temp;
}

//---------------------------------------------------------------

void launchKernel( cudaPitchedPtr aDev, cudaPitchedPtr bDev, cudaPitchedPtr cDev, int aH, int aW, int bW )
{
   dim3 block = dim3 ( BLOCK_SIZE, BLOCK_SIZE );
   dim3 grid = dim3 ( aH / BLOCK_SIZE, bW / BLOCK_SIZE );
   SAFE_KERNEL_CALL( ( sum_kernel<<< grid, block >>>( aDev, bDev, cDev, aH, aW, bW ) ) );
}

//---------------------------------------------------------------

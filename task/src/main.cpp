#include <iostream>

#include "defs.h"
#include "helper.h"
#include "parparser.h"

//---------------------------------------------------------------

void launchPartOneKernel( cudaPitchedPtr aDev, cudaPitchedPtr bDev, cudaPitchedPtr cDev, int aH, int aW, int bW );
void launchPartTwoKernel( cudaPitchedPtr aDev, cudaPitchedPtr bDev, cudaPitchedPtr cDev, int aH, int aW, int bW );

//---------------------------------------------------------------

int main( int argc, char** argv )
{
    parparser arguments( argc, argv );
    int mataH = arguments.get( "ah" ).asInt(0);
    int mataW = arguments.get( "aw" ).asInt(0);
    int matbW = arguments.get( "bw" ).asInt(0);
    bool isCheck = arguments.get( "check" ).asBool( false );

    if ( mataH <= 0 || mataW <= 0 || matbW <= 0 )
    {
        std::cout << "Incorrect parameters\r\n";
        exit(0);
    }

    int device = selectCUDADevice();
    if ( device == -1 )
    {
        std::cout << "Not found suitable device\r\n";
        exit(0);
    }
    SAFE_CALL( cudaSetDevice( device ) );

    //-----------------------------------------------------------

    cudaPitchedPtr aHost = make_cudaPitchedPtr( new MATRIX_TYPE[ mataH * mataW ], mataW * sizeof( MATRIX_TYPE ), mataH, mataW );
    cudaPitchedPtr bHost = make_cudaPitchedPtr( new MATRIX_TYPE[ matbW * mataW ], matbW * sizeof( MATRIX_TYPE ), mataH, matbW );
    cudaPitchedPtr cHost = { 0, 0, 0, 0 };
    cudaPitchedPtr cHostFromDev = make_cudaPitchedPtr( new MATRIX_TYPE[ mataH * matbW ], matbW * sizeof( MATRIX_TYPE ), mataH, matbW );
   
    srand( 0U );
    initializeRandomArray< MATRIX_TYPE >( aHost );
    initializeRandomArray< MATRIX_TYPE >( bHost );

    if ( isCheck )
    {
        cHost = make_cudaPitchedPtr( new MATRIX_TYPE[ mataH * matbW ], matbW * sizeof( MATRIX_TYPE ), mataH, matbW );
        hostMul< MATRIX_TYPE >( aHost, bHost, &cHost );
    }

    cudaPitchedPtr aDev;
    cudaPitchedPtr bDev;
    cudaPitchedPtr cDev;
    
    #ifdef PARTONE

        SAFE_CALL( cudaMalloc( &aDev.ptr, mataH * mataW ) );
        SAFE_CALL( cudaMalloc( &bDev.ptr, matbW * mataW ) );
        SAFE_CALL( cudaMalloc( &cDev.ptr, mataH * matbW ) );
        aDev.pitch = mataW * sizeof( MATRIX_TYPE );
        bDev.pitch = matbW * sizeof( MATRIX_TYPE );
        cDev.pitch = matbW * sizeof( MATRIX_TYPE );

        SAFE_CALL( cudaMemcpy( aDev.ptr, aHost.ptr, mataH * mataW, cudaMemcpyHostToDevice ) );
        SAFE_CALL( cudaMemcpy( bDev.ptr, bHost.ptr, matbW * mataW, cudaMemcpyHostToDevice ) );

        launchPartOneKernel( aDev, bDev, cDev, mataH, mataW, matbW );

        SAFE_CALL( cudaMemcpy( cHostFromDev.ptr, cDev.ptr, mataH * matbW, cudaMemcpyDeviceToHost ) );
    #endif
         
    #ifdef PARTTWO
        SAFE_CALL( cudaMallocPitch( &aDev.ptr, &aDev.pitch, mataW * sizeof( MATRIX_TYPE ) , mataH ) );
        SAFE_CALL( cudaMallocPitch( &bDev.ptr, &bDev.pitch, matbW * sizeof( MATRIX_TYPE ) , mataW ) );
        SAFE_CALL( cudaMallocPitch( &cDev.ptr, &cDev.pitch, matbW * sizeof( MATRIX_TYPE ) , mataH ) );

        SAFE_CALL( cudaMemcpy2D( aDev.ptr, aDev.pitch, aHost.ptr, aHost.pitch, mataW * sizeof( MATRIX_TYPE ), mataH, cudaMemcpyHostToDevice ) );
        SAFE_CALL( cudaMemcpy2D( bDev.ptr, bDev.pitch, bHost.ptr, bHost.pitch, matbW * sizeof( MATRIX_TYPE ), mataW, cudaMemcpyHostToDevice ) );

        launchPartTwoKernel( aDev, bDev, cDev, mataH, mataW, matbW );

        SAFE_CALL( cudaMemcpy2D( cHostFromDev.ptr, cHostFromDev.pitch, cDev.ptr, cDev.pitch, matbW * sizeof( MATRIX_TYPE ), mataH, cudaMemcpyDeviceToHost ) );
    #endif   

    //-----------------------------------------------------------

    if ( isCheck )
        if ( !cmpMatrix( cHost, cHostFromDev, EPS ) )
            std::cout << "CHECK: unequal matrices\r\n";

    //-----------------------------------------------------------

    delete[] ( MATRIX_TYPE* )aHost.ptr;
    delete[] ( MATRIX_TYPE* )bHost.ptr;
    delete[] ( MATRIX_TYPE* )cHost.ptr;
    delete[] ( MATRIX_TYPE* )cHostFromDev.ptr;

    SAFE_CALL( cudaFree( aDev.ptr ) );
    SAFE_CALL( cudaFree( bDev.ptr ) );
    SAFE_CALL( cudaFree( cDev.ptr ) );

    //-----------------------------------------------------------
    return 0;
}

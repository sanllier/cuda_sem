#include <iostream>

#include "defs.h"
#include "helper.h"
#include "parparser.h"

//---------------------------------------------------------------

void launchKernel( cudaPitchedPtr aDev, cudaPitchedPtr bDev, cudaPitchedPtr cDev, int aH, int aW, int bW );

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

    MATRIX_TYPE* aHost = new MATRIX_TYPE[ mataH * mataW ];
    MATRIX_TYPE* bHost = new MATRIX_TYPE[ matbW * mataW ];
    MATRIX_TYPE* cHost = 0;
    MATRIX_TYPE* cHostFromDev = new MATRIX_TYPE[ mataH * matbW ];

    srand( 0U );
    initializeRandomArray( aHost, mataH * mataW );
    initializeRandomArray( bHost, matbW * mataW );

    if ( isCheck )
    {
        cHost = new MATRIX_TYPE[ mataH * matbW ];
        hostMul( aHost, bHost, cHost, mataH, mataW, matbW );
    }

    cudaPitchedPtr aPitchedPtr = make_cudaPitchedPtr( 0, 0, mataW, mataH );
    cudaPitchedPtr bPitchedPtr = make_cudaPitchedPtr( 0, 0, matbW, mataW );
    cudaPitchedPtr cPitchedPtr = make_cudaPitchedPtr( 0, 0, matbW, mataH );

    SAFE_CALL( cudaMallocPitch( &aPitchedPtr.ptr, &aPitchedPtr.pitch, mataW * sizeof( MATRIX_TYPE ) , mataH ) );
    SAFE_CALL( cudaMallocPitch( &bPitchedPtr.ptr, &bPitchedPtr.pitch, matbW * sizeof( MATRIX_TYPE ) , mataW ) );
    SAFE_CALL( cudaMallocPitch( &cPitchedPtr.ptr, &cPitchedPtr.pitch, matbW * sizeof( MATRIX_TYPE ) , mataH ) );

    SAFE_CALL( cudaMemcpy2D( aPitchedPtr.ptr, aPitchedPtr.pitch, aHost, 0, mataW * sizeof( MATRIX_TYPE ), mataH, cudaMemcpyHostToDevice ) );
    SAFE_CALL( cudaMemcpy2D( bPitchedPtr.ptr, bPitchedPtr.pitch, bHost, 0, matbW * sizeof( MATRIX_TYPE ), mataW, cudaMemcpyHostToDevice ) );

    launchKernel( aPitchedPtr, bPitchedPtr, cPitchedPtr, mataH, mataW, matbW );

    SAFE_CALL( cudaMemcpy2D( cHostFromDev, 0, cPitchedPtr.ptr, cPitchedPtr.pitch, matbW * sizeof( MATRIX_TYPE ), mataH, cudaMemcpyDeviceToHost ) );

    //-----------------------------------------------------------

    if ( isCheck )
        if ( !cmpMatrix( cHost, cHostFromDev, mataH, matbW ) )
            std::cout << "CHECK: unequal matrices\r\n";

    //-----------------------------------------------------------

    delete[] aHost;
    delete[] bHost;
    delete[] cHost;
    delete[] cHostFromDev;

    SAFE_CALL( cudaFree( aPitchedPtr.ptr ) );
    SAFE_CALL( cudaFree( bPitchedPtr.ptr ) );
    SAFE_CALL( cudaFree( cPitchedPtr.ptr ) );

    //-----------------------------------------------------------
    return 0;
}

#include <iostream>
#include <cstring>
#include <ctime>

#include "defs.h"
#include "helper.h"
#include "parparser.h"

#define PARTONE

//---------------------------------------------------------------

void launchPartOneKernel( cudaPitchedPtr aDev, cudaPitchedPtr bDev, cudaPitchedPtr cDev, int aH, int aW, int bW, int blockH, int blockW );
void launchPartTwoKernel( cudaPitchedPtr aDev, cudaPitchedPtr bDev, cudaPitchedPtr cDev, int aH, int aW, int bW, int blockH, int blockW );
void launchPartThreeKernel( cudaPitchedPtr aDev, cudaPitchedPtr bDev, cudaPitchedPtr cDev, int aH, int aW, int bW, int blockH, int blockW );

//---------------------------------------------------------------

int main( int argc, char** argv )
{
    parparser arguments( argc, argv );
    int mataH = arguments.get( "ah" ).asInt(0);
    int mataW = arguments.get( "aw" ).asInt(0);
    int matbW = arguments.get( "bw" ).asInt(0);
    bool isCheck = arguments.get( "check" ).asBool( false );
    int blockH = arguments.get( "blockh" ).asInt( 2 );
    int blockW = arguments.get( "blockW" ).asInt( 2 );

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

    #if defined( PARTONE ) || defined( PARTTWO )
        const int aHOverhead = 0;
        const int aWOverhead = 0;
        const int bWOverhead = 0;
    #endif
    #ifdef PARTTHREE
        const int aHOverhead = blockH - mataH % blockH;
        const int aWOverhead = blockW - mataW % blockW;
        const int bWOverhead = blockW - matbW % blockW;
    #endif

    cudaPitchedPtr aHost = make_cudaPitchedPtr( new MATRIX_TYPE[ ( mataH + aHOverhead ) * ( mataW + aWOverhead ) ], \
        ( mataW + aWOverhead ) * sizeof( MATRIX_TYPE ), mataW, mataH );
    cudaPitchedPtr bHost = make_cudaPitchedPtr( new MATRIX_TYPE[ ( matbW + bWOverhead ) * ( mataW + aWOverhead ) ], \
        ( matbW + bWOverhead ) * sizeof( MATRIX_TYPE ), matbW, mataH );
    cudaPitchedPtr cHost = { 0, 0, 0, 0 };
    cudaPitchedPtr cHostFromDev = make_cudaPitchedPtr( new MATRIX_TYPE[ ( mataH + aHOverhead ) * ( matbW + bWOverhead ) ], \
        ( matbW + bWOverhead ) * sizeof( MATRIX_TYPE ), matbW, mataH );

    std::memset( aHost.ptr, 0, aHost.pitch * ( mataH + aHOverhead ) );
    std::memset( bHost.ptr, 0, bHost.pitch * ( mataW + aWOverhead ) );
    std::memset( cHostFromDev.ptr, 0, cHostFromDev.pitch * ( mataH + aHOverhead ) );
    
    srand( ( unsigned )time( 0U ) );
    initializeRandomArray< MATRIX_TYPE >( aHost );
    initializeRandomArray< MATRIX_TYPE >( bHost );

    if ( isCheck )
    {
        cHost = make_cudaPitchedPtr( new MATRIX_TYPE[ ( mataH + aHOverhead ) * ( matbW + bWOverhead ) ], \
            ( matbW + bWOverhead ) * sizeof( MATRIX_TYPE ), matbW, mataH );
        std::memset( cHost.ptr, 0, cHost.pitch * ( mataH + aHOverhead ) );
        hostMul< MATRIX_TYPE >( aHost, bHost, &cHost );
    }

    cudaPitchedPtr aDev;
    cudaPitchedPtr bDev;
    cudaPitchedPtr cDev;
    
    #ifdef PARTONE
        SAFE_CALL( cudaMalloc( &aDev.ptr, mataH * mataW * sizeof( MATRIX_TYPE ) ) );
        SAFE_CALL( cudaMalloc( &bDev.ptr, matbW * mataW * sizeof( MATRIX_TYPE ) ) );
        SAFE_CALL( cudaMalloc( &cDev.ptr, mataH * matbW * sizeof( MATRIX_TYPE ) ) );
        aDev.pitch = mataW * sizeof( MATRIX_TYPE );
        bDev.pitch = matbW * sizeof( MATRIX_TYPE );
        cDev.pitch = matbW * sizeof( MATRIX_TYPE );

        SAFE_CALL( cudaMemcpy( aDev.ptr, aHost.ptr, mataH * mataW * sizeof( MATRIX_TYPE ), cudaMemcpyHostToDevice ) );
        SAFE_CALL( cudaMemcpy( bDev.ptr, bHost.ptr, matbW * mataW * sizeof( MATRIX_TYPE ), cudaMemcpyHostToDevice ) );

        launchPartOneKernel( aDev, bDev, cDev, mataH, mataW, matbW, blockH, blockW );

        SAFE_CALL( cudaMemcpy( cHostFromDev.ptr, cDev.ptr, mataH * matbW, cudaMemcpyDeviceToHost ) );
    #endif

    #ifdef PARTTWO
        SAFE_CALL( cudaMallocPitch( &aDev.ptr, &aDev.pitch, mataW * sizeof( MATRIX_TYPE ), mataH ) );
        SAFE_CALL( cudaMallocPitch( &bDev.ptr, &bDev.pitch, matbW * sizeof( MATRIX_TYPE ), mataW ) );
        SAFE_CALL( cudaMallocPitch( &cDev.ptr, &cDev.pitch, matbW * sizeof( MATRIX_TYPE ), mataH ) );

        SAFE_CALL( cudaMemcpy2D( aDev.ptr, aDev.pitch, aHost.ptr, aHost.pitch, mataW * sizeof( MATRIX_TYPE ), mataH, cudaMemcpyHostToDevice ) );
        SAFE_CALL( cudaMemcpy2D( bDev.ptr, bDev.pitch, bHost.ptr, bHost.pitch, matbW * sizeof( MATRIX_TYPE ), mataW, cudaMemcpyHostToDevice ) );

        launchPartTwoKernel( aDev, bDev, cDev, mataH, mataW, matbW, blockH, blockW );

        SAFE_CALL( cudaMemcpy2D( cHostFromDev.ptr, cHostFromDev.pitch, cDev.ptr, cDev.pitch, matbW * sizeof( MATRIX_TYPE ), mataH, cudaMemcpyDeviceToHost ) );
    #endif 

    #ifdef PARTTHREE
        SAFE_CALL( cudaMallocPitch( &aDev.ptr, &aDev.pitch, ( mataW + aWOverhead ) * sizeof( MATRIX_TYPE ), mataH + aHOverhead ) );
        SAFE_CALL( cudaMallocPitch( &bDev.ptr, &bDev.pitch, ( matbW + bWOverhead ) * sizeof( MATRIX_TYPE ), mataW + aWOverhead ) );
        SAFE_CALL( cudaMallocPitch( &cDev.ptr, &cDev.pitch, ( matbW + bWOverhead ) * sizeof( MATRIX_TYPE ), mataH + aHOverhead ) );

        SAFE_CALL( cudaMemcpy2D( aDev.ptr, aDev.pitch, aHost.ptr, aHost.pitch, ( mataW + aWOverhead ) * sizeof( MATRIX_TYPE ), \
            mataH + aHOverhead, cudaMemcpyHostToDevice ) );
        SAFE_CALL( cudaMemcpy2D( bDev.ptr, bDev.pitch, bHost.ptr, bHost.pitch, ( matbW + bWOverhead ) * sizeof( MATRIX_TYPE ), \
            mataW + aWOverhead, cudaMemcpyHostToDevice ) );

        launchPartThreeKernel( aDev, bDev, cDev, mataH + aHOverhead, mataW + aWOverhead, matbW + bWOverhead, blockH, blockW );

        SAFE_CALL( cudaMemcpy2D( cHostFromDev.ptr, cHostFromDev.pitch, cDev.ptr, cDev.pitch, ( matbW + bWOverhead ) * sizeof( MATRIX_TYPE ), \
            mataH + aHOverhead, cudaMemcpyDeviceToHost ) );
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

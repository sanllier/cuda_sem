#ifndef HELPER_H
#define HELPER_H

#include "cuda_runtime.h"
#include "cudaErrorHandler.h"

#include <iostream>
#include <cstdlib>

//---------------------------------------------------------------

int selectCUDADevice()
{
   int deviceCount = 0;
   int suitableDevice = -1;
   cudaDeviceProp devProp;  

   SAFE_CALL( cudaGetDeviceCount( &deviceCount ) );
   std::cout << "Found " << deviceCount << " devices: \r\n";   
   for ( int device = 0; device < deviceCount; ++device )
   {
       SAFE_CALL( cudaGetDeviceProperties ( &devProp, device ) );
       
       std::cout << "Device: " << device                                               << std::endl;
       std::cout << "   Compute capability: " << devProp.major << "." << devProp.minor << std::endl;
       std::cout << "   Name: " << devProp.name                                        << std::endl;
       std::cout << "   Total Global Memory: " << devProp.totalGlobalMem               << std::endl;
       std::cout << "   Shared Memory Per Block: " << devProp.sharedMemPerBlock        << std::endl;
       
       if( devProp.major >= 2 )
          suitableDevice = device ;
   }
   return suitableDevice;
}

//---------------------------------------------------------------

//NOTE: srand before this function needed
template< typename T >
void initializeRandomArray( T* arr, size_t len )
{
    static int MAX_VAL = 32768; 
    for ( size_t i = 0; i < len; ++i )
        arr[i] = T( rand() % MAX_VAL );
}

//---------------------------------------------------------------

template< typename T >
bool hostMul( const T* aMat, const T* bMat, T* cMat, int aH, int aW, int bW )
{
    if ( aH <= 0 || aW <= 0 || bW <= 0 || !aMat || !bMat || !cMat )
        return false;

    for ( int i = 0; i < aH; ++i )
    {
        for ( int q = 0; q < bW; ++q )
        {
            int cMatPos = i * bW + q;
            cMat[ cMatPos ] = T(0);

            for ( int k = 0; k < aW )
              cMat[ cMatPos ] += aMat[ i * aW + k ] * bMat[ k * bW + q ];
        }
    }
    return true;
}

//---------------------------------------------------------------

template< typename T >
bool printMatrix( std::ostream& oStr, const T* mat, int h, int w )
{
    if ( !mat || h <= 0 || w <= 0 )
        return false;

    for ( int i = 0; i < h; ++i )
        for ( int q = 0; q < w; ++q )
            oStr << mat[ i * w + q ];

    return true;
}

//---------------------------------------------------------------

template< typename T >
bool cmpMatrix( const T* aMat, const T* bMat, int h, int w, T eps = T(0) )
{
    for ( int i = 0; i < h * w; ++i )
        if ( fabs( aMat[i] - bMat[i] ) >= eps )
            return false;

    return true;
}

//---------------------------------------------------------------

#endif

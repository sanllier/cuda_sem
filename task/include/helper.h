#ifndef HELPER_H
#define HELPER_H

#include "cuda_runtime.h"
#include "cudaErrorHandler.h"

#include "defs.h"

#include <iostream>
#include <cstdlib>
#include <cmath>

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
   if ( suitableDevice > -1 )
       std::cout << "Suitable device: " << suitableDevice << "\r\n";
   return suitableDevice;
}

//---------------------------------------------------------------

//NOTE: srand before this function needed
template< typename T >
void initializeRandomArray( cudaPitchedPtr& arr )
{
    for ( int i = 0; i < arr.xsize; ++i )
        for ( int q = 0; q < arr.ysize; ++q )
            get_elem( arr, i, q ) =  T( rand() % MAX_RAND_VAL );
}

//---------------------------------------------------------------

template< typename T >
bool hostMul( const cudaPitchedPtr& aMat, const cudaPitchedPtr& bMat, cudaPitchedPtr* cMat )
{
    if ( aMat.xsize <= 0 || aMat.ysize <= 0 || bMat.xsize <= 0 || bMat.ysize <= 0 || aMat.ysize != bMat.xsize || !cMat )
        return false;

    for ( int i = 0; i < aMat.xsize; ++i )
    {
        for ( int q = 0; q < bMat.ysize; ++q )
        {
            T temp = T(0);
            for ( int k = 0; k < aMat.ysize; ++k )
                temp += get_elem( aMat, i, k ) * get_elem( bMat, k, q );

            get_elem( (*cMat), i, q ) = temp;
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
    {
        for ( int q = 0; q < w; ++q )
            oStr << mat[ i * w + q ] << " ";

        oStr << "\r\n";
    }

    return true;
}

//---------------------------------------------------------------

template< typename T >
bool cmpMatrix( const cudaPitchedPtr& aMat, const cudaPitchedPtr& bMat, T eps = T(0) )
{
    if ( aMat.xsize != bMat.xsize || aMat.ysize != bMat.ysize )
    {
        std::cout << "Unequal sizes\r\n";
        return false;
    }

    for ( int i = 0; i < aMat.xsize; ++i )
    {
        for ( int q = 0; q < aMat.ysize; ++q )
            if ( fabs( get_elem( aMat, i, q ) - get_elem( bMat, i, q ) ) < eps )
            {
                std::cout << "| " << get_elem( aMat, i, q ) << " - " << get_elem( bMat, i, q ) << " | = " \
                          << fabs( get_elem( aMat, i, q ) - get_elem( bMat, i, q ) ) << " > " << eps << "\r\n"; 
                return false;
            }
    }


    return true;
}

//---------------------------------------------------------------

#endif

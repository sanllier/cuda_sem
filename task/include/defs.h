#ifndef DEFS_H
#define DEFS_H

//---------------------------------------------------------------

#define MATRIX_TYPE int
#define BLOCK_SIZE 2
#define MAX_RAND_VAL RAND_MAX
#define EPS 0

//---------------------------------------------------------------

#define get_elem( _pptr_, _row_, _col_ ) ( *( ( MATRIX_TYPE* )( ( char* )_pptr_.ptr + _row_ * _pptr_.pitch ) + _col_ ) )

//---------------------------------------------------------------

#endif

#ifndef DEFS_H
#define DEFS_H

//---------------------------------------------------------------

#define MATRIX_TYPE int
#define MAX_RAND_VAL 3
#define EPS 0

//---------------------------------------------------------------

#define get_elem( _pptr_, _row_, _col_ ) ( *( ( MATRIX_TYPE* )( ( char* )_pptr_.ptr + _row_ * _pptr_.pitch ) + _col_ ) )

//---------------------------------------------------------------

#endif

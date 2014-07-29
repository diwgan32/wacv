/*
 * MATLAB Compiler: 5.0 (R2013b)
 * Date: Mon Jul 28 09:59:40 2014
 * Arguments: "-B" "macro_default" "-B" "csharedlib:construct_dict" "-W"
 * "lib:construct_dict" "-T" "link:lib" "construct_dict.m" 
 */

#ifndef __construct_dict_h
#define __construct_dict_h 1

#if defined(__cplusplus) && !defined(mclmcrrt_h) && defined(__linux__)
#  pragma implementation "mclmcrrt.h"
#endif
#include "mclmcrrt.h"
#ifdef __cplusplus
extern "C" {
#endif

#if defined(__SUNPRO_CC)
/* Solaris shared libraries use __global, rather than mapfiles
 * to define the API exported from a shared library. __global is
 * only necessary when building the library -- files including
 * this header file to use the library do not need the __global
 * declaration; hence the EXPORTING_<library> logic.
 */

#ifdef EXPORTING_construct_dict
#define PUBLIC_construct_dict_C_API __global
#else
#define PUBLIC_construct_dict_C_API /* No import statement needed. */
#endif

#define LIB_construct_dict_C_API PUBLIC_construct_dict_C_API

#elif defined(_HPUX_SOURCE)

#ifdef EXPORTING_construct_dict
#define PUBLIC_construct_dict_C_API __declspec(dllexport)
#else
#define PUBLIC_construct_dict_C_API __declspec(dllimport)
#endif

#define LIB_construct_dict_C_API PUBLIC_construct_dict_C_API


#else

#define LIB_construct_dict_C_API

#endif

/* This symbol is defined in shared libraries. Define it here
 * (to nothing) in case this isn't a shared library. 
 */
#ifndef LIB_construct_dict_C_API 
#define LIB_construct_dict_C_API /* No special import/export declaration */
#endif

extern LIB_construct_dict_C_API 
bool MW_CALL_CONV construct_dictInitializeWithHandlers(
       mclOutputHandlerFcn error_handler, 
       mclOutputHandlerFcn print_handler);

extern LIB_construct_dict_C_API 
bool MW_CALL_CONV construct_dictInitialize(void);

extern LIB_construct_dict_C_API 
void MW_CALL_CONV construct_dictTerminate(void);



extern LIB_construct_dict_C_API 
void MW_CALL_CONV construct_dictPrintStackTrace(void);

extern LIB_construct_dict_C_API 
bool MW_CALL_CONV mlxConstruct_dict(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);



extern LIB_construct_dict_C_API bool MW_CALL_CONV mlfConstruct_dict(int nargout, mxArray** printD, mxArray** printPINVD, mxArray* SubjectData, mxArray* Segments, mxArray* params);

#ifdef __cplusplus
}
#endif
#endif

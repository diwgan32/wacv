/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** utils.h
** Set of tools to facilitate coding. Includes utilities to 
** convert from int to string, and reading data from a binary file
**
** Author: Diwakar Ganesan
** -------------------------------------------------------------------------*/


#ifndef UTILS_H
#define UTILS_H

Mat readBin(const char * filename, int numRows, int numCols);

string itos(int a);

bool wayToSort(int i, int j);

#endif

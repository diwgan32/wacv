/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** Group.h
** Struct that facilitates the formation of partitions. 
** Used primarily by segments.h.
**
** Author: Diwakar Ganesan
** -------------------------------------------------------------------------*/
#ifndef GROUP_H
#define GROUP_H
#include "includes.h"
using namespace cv;
typedef struct{
	vector<int> * segments;
	vector<Mat> centers;
	vector<int> count;
} Groupings;

#endif

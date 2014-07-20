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
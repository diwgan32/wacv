/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** mp_test.h
** Functions to measure the similarity between two videos to 
** generate similarity matrices.
** 
** Author: Diwakar Ganesan
** -------------------------------------------------------------------------*/

#ifndef MP_TEST
#define MP_TEST

using namespace cv;
using namespace std;


double project(Mat img, Mat D, Mat pinvD);
double * image_test(Mat img, vector<Mat> D, vector<Mat> pinvD);
double  image_test(Mat img, Mat D, Mat pinvD);	
#endif

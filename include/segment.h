/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** segment.h
** Functions to partition frames into groups
**
** Author: Diwakar Ganesan
** -------------------------------------------------------------------------*/

#ifndef SEGMENT_H
#define SEGMENT_H
#include "Group.h"

Groupings myKMeans(cv::Mat Y, int k);

Groupings seg(Mat ImgData, int k);

Groupings addFrame(Groupings g, Mat frame, int frameid, int k);

double scatter(Mat data, Groupings g, int k);

Groupings reset(Mat data, Groupings g, int k);

vector<Mat> splitBySegment(Mat data, Groupings g);

#endif

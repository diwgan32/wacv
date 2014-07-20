#ifndef SEGMENT_H
#define SEGMENT_H
#include "Group.h"
Groupings myKMeans(cv::Mat Y, int k);
Groupings seg(Mat ImgData, int k);
Groupings addFrame(Groupings g, Mat frame, int frameid, int k);
double scatter(Mat data, Groupings g, int k);
Groupings createCenters(Groupings g, Mat data, int k);
Groupings reset(Mat data, Groupings g, int k);
Groupings seg1(Mat ImgData, int k);
vector<Mat> splitBySegment(Mat data, Groupings g);
#endif
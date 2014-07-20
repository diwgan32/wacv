#include "includes.h"

using namespace cv;
double project(Mat img, Mat D, Mat pinvD){
	Mat feature;
	feature = pinvD*img;
	Mat temp;
	Mat temp1 = img-(D*feature);
	pow(img-(D*feature), 2.0, temp);
	pow(temp, 2.0, temp);
	double res = sum(temp)[0];

	return res;
}
Mat image_test(Mat img, vector<Mat> D, vector<Mat> pinvD){
	Mat dist = Mat(img.cols, D.size(), CV_64F);
	Mat temp;
	double res;
	Mat temp1(400, 1, CV_64F);
	for(int i = 0; i<img.cols; i++){
		for(int j = 0; j<D.size(); j++){
			temp1 = img.col(i);

			res = project(temp1, D.at(j), pinvD.at(j));

			pow(res, 2.0);
			dist.at<double>(i, j) = res;
		}
	}

	Mat I;
	cv::sortIdx(dist, I, CV_SORT_ASCENDING + CV_SORT_EVERY_ROW);
	return I.col(0);
}


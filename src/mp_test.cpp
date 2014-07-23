#include "includes.h"

using namespace cv;
double project(Mat img, Mat D, Mat pinvD){
	Mat feature;

	feature = pinvD*img;
	Mat temp;
	Mat temp1 = img-(D*feature);
	pow(temp1, 2.0, temp);

	double res = sum(temp)[0];

	return res;
}
double* image_test(Mat img, vector<Mat> D, vector<Mat> pinvD){
	Mat dist = Mat(img.cols, D.size(), CV_64F, 0.0);
	Mat temp;
	double res;
	double minres = 0;
	Mat temp1(400, 1, CV_64F);
	double * sim_mat = new double[D.size()];
	for(int j = 0; j<D.size(); j++){
		for(int i = 0; i<img.cols; i++){

			temp1 = img.col(i);
			res = project(temp1, D.at(j), pinvD.at(j));
			if(res < minres){
				minres = res;
			}

		}
		sim_mat[j] = minres;
		minres = 0;
	}
	
	return sim_mat;
}


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

	Mat temp;
	double res;
	double minres = DBL_MAX;
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
		minres = DBL_MAX;
	}

	return sim_mat;
}

double image_test(Mat img, Mat D, Mat pinvD){
	Mat temp;
	double res;
	double minres = DBL_MAX;
	Mat temp1(400, 1, CV_64F);
	double sim;
	for(int i = 0; i<img.cols; i++){

		temp1 = img.col(i);
		res = project(temp1, D, pinvD);
		if(res < minres){
			minres = res;
		}

	}
	sim = minres;
	return sim;
}


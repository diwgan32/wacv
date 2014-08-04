#include "includes.h"

using namespace cv;

/*
 * Function to calculate residuals between
 * query frames and a dictionary.
*/
double project(Mat img, Mat D, Mat pinvD){
	Mat feature;

	feature = pinvD*img;
	Mat temp;
	Mat temp1 = img-(D*feature);
	pow(temp1, 2.0, temp);

	double res = sum(temp)[0];

	return res;
}

/* 
 * Legacy function that returns an array of similarity
 * values to be incorporated into similarity matrix. Used
 * when only 10 subjects were trained. When more test
 * data was incorporated, the function was modified to
 * only return one value. See below..
*/

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

	//Loop through all frames and find minimum residual.
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

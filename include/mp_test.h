#ifndef MP_TEST
#define MP_TEST

using namespace cv;
using namespace std;


double project(Mat img, Mat D, Mat pinvD);
Mat image_test(Mat img, vector<Mat> D, vector<Mat> pinvD);
	
#endif
#include "includes.h"

using namespace cv;

Mat readBin(const char * filename, int numRows, int numCols){
	std::ifstream ifs(filename, std::ios::binary);
	Mat result(numRows, 1, CV_64F);
	Mat temp(numRows, 1, CV_64F);
	bool flag = false;
	double val;
	for(int i = 0; i<numCols; i++){
		for(int j = 0; j<numRows; j++){
			if(i==0){
				ifs.read(reinterpret_cast<char*> (&result.at<double>(j, 0)) , sizeof val);
			
			}else{
					flag = true;
				ifs.read(reinterpret_cast<char*> (&temp.at<double>(j, 0)) , sizeof val);
				
			}
			
		}
		
		if(flag){
		hconcat(result, temp, result);
		}
	}
	ifs.close();
	return result;
}

string itos(int a){

	std::stringstream ss;
	ss << a;
	string str = ss.str();
	return str;
}

bool wayToSort(int i, int j) { return i > j; }


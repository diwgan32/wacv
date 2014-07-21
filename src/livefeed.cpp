// WARNING: this sample is under construction! Use it on your own risk.
#if defined _MSC_VER && _MSC_VER >= 1400
#pragma warning(disable : 4100)
#endif



#include "includes.h"
#include "mp_test.h"
#include "segment.h"
#include "utils.h"
#define NUM_DICT 15
#define NUM_SUBJECTS 15
#define COLS 3
#define ROWS 400
using namespace std;
using namespace cv;
using namespace cv::gpu;

template<class T>
void convertAndResize(const T& src, T& gray, T& resized, double scale)
{
	if (src.channels() == 3)
	{
		cvtColor( src, gray, CV_BGR2GRAY );
	}
	else
	{
		gray = src;
	}

	Size sz(cvRound(gray.cols * scale), cvRound(gray.rows * scale));

	if (scale != 1)
	{
		resize(gray, resized, sz);
	}
	else
	{
		resized = gray;
	}
}


int main(int argc, const char *argv[])
{

	ifstream fin;
	Mat ImgData;

	cout << "Reading Data....";
	ImgData = readBin("DataFiles\\HONDA_DATA.bin", 400, 100);
	cout << "Done" << endl;

	fin.open("DataFiles\\subjectFrameNums.txt", std::ifstream::in);
	cout << "Train or recog?: ";
	int x;
	cin >> x;
	cout << endl;
	while(x!=3){
		int pos = 0;
		ofstream fout;
		int numSegments = 3;



		switch (x){

		case 0:


			for(int z = 0; z<15; z++){
				// 249 306 163 240 263
				MATFile *pmat;
				mxArray *pa1;
				int size = 0;
				fin >> size;
				cout << size <<" ";
				size = size-1;

				Mat portion = ImgData.colRange(Range(pos, pos+100));
				pos+=size;
				size = 100;
				Groupings g = seg1(portion, numSegments);

				int maxnum = 0;
				for(int j = 0; j<numSegments; j++){

					if(g.segments[j].size()>maxnum){
						maxnum = g.segments[j].size();
					}
				}
				pa1 = mxCreateDoubleMatrix(numSegments, maxnum, mxREAL);
				string filename = "Segments\\"+itos(z)+".mat";
				pmat = matOpen(filename.c_str(), "w");

				double * data2;
				int count = 0;
				data2 = new double[numSegments*maxnum];
				for(int j = 0; j<maxnum; j++){
					for(int k = 0; k<numSegments;  k++){
						if(j>=g.segments[k].size()){
							data2[count] = -1;
						}
						else{
							data2[count] = g.segments[k].at(j);
						}
						count++;
					}
				}

				memcpy((void *)(mxGetPr(pa1)), data2, sizeof(data2)*maxnum*numSegments);
				int status = matPutVariable(pmat, "Segments", pa1);
				if (status != 0) {

					printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
					return(EXIT_FAILURE);
				}  

				mxDestroyArray(pa1);
				matClose(pmat);



				pa1 = mxCreateDoubleMatrix(400, size, mxREAL);
				filename = "Subjects\\"+itos(z)+".mat";
				pmat = matOpen(filename.c_str(), "w");

				double * data1;
				data1 = new double[size*400];
				count = 0;
				for(int i = 0; i<size; i++){
					for(int j = 0; j<400; j++){
						data1[count] = portion.at<double>(j, i);
						count++;
					}

				}


				memcpy((void *)(mxGetPr(pa1)), data1, sizeof(data1)*400*size);
				status = matPutVariable(pmat, "SubjectData", pa1);
				if (status != 0) {

					printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
					return(EXIT_FAILURE);
				}  

				mxDestroyArray(pa1);
				matClose(pmat);

				cout  << z << endl;
				delete data1;
				delete data2;
			}
			fin.close();
			break;

		case 1:

			pos = 0;


			for(int z = 0; z<15; z++){
				// 249 183 292 306 249
				MATFile *pmat;
				mxArray *pa1;
				int size = 0;
				fin >> size;
				size = size-1;

				Mat portion = ImgData.colRange(Range(pos+100, pos+150));
				pos+=size;
				size = 50;
				Groupings g = seg1(portion, numSegments);
				vector<Mat> split = splitBySegment(portion, g);
				//cout << portion << endl;
				ifstream ifs("Dictionaries\\dict.bin", ios::binary);
				vector<Mat> dict;
				for(int i = 0; i<NUM_DICT; i++){
					dict.push_back(Mat (ROWS, COLS, CV_64F));
				}
				double val;
				for(int i = 0; i<COLS*NUM_DICT; i++){
					for(int j = 0; j<ROWS; j++){
						ifs.read(reinterpret_cast<char*> (&dict.at(i/COLS).at<double>(j, i%COLS)) , sizeof val);
					}

				}
				ifs.close();

				ifstream ifs1("Dictionaries\\pinvDict.bin", ios::binary);
				vector<Mat> pinvD;
				for(int i = 0; i<NUM_DICT; i++){
					pinvD.push_back(Mat(COLS, ROWS, CV_64F));
				}
				for(int i = 0; i<ROWS*NUM_DICT; i++){
					for(int j = 0; j<COLS; j++){
						ifs1.read(reinterpret_cast<char*> (&pinvD.at(i/ROWS).at<double>(j, i%ROWS)) , sizeof val);
					}
				}
				ifs1.close();
				vector<int> scores;
				scores.reserve(NUM_SUBJECTS);
				int max = 0;
				int id = 0;
				for(int i = 0; i<NUM_SUBJECTS; i++){
					scores.push_back(0);
				}
				for(int i = 0; i<numSegments; i++){
					Mat labels = image_test(split.at(i), dict, pinvD);


					for(int j = 0; j<labels.rows; j++){
						scores.at(labels.at<int>(j, 0)) += 1;
					}

					max = 0;
					id = 0;
					for(int j = 0; j<NUM_SUBJECTS; j++){
						if(scores.at(j) > max){
							max = scores.at(j);
							id = j;
						}
					}
				}
				sort(scores.begin(), scores.end(), wayToSort);
				cout << "Subject: #"<<id << endl;

			}
			fin.close();
			break;
		}
		cout << "Train or recog?: ";
		int x;
		cin >> x;
		cout << endl;
	}


	return 0;
}

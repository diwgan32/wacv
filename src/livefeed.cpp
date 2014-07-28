// WARNING: this sample is under construction! Use it on your own risk.
#if defined _MSC_VER && _MSC_VER >= 1400
#pragma warning(disable : 4100)
#endif

#include "includes.h"
#include "Group.h"
#include "mp_test.h"
#include "segment.h"
#include "utils.h"
#include "dirent.h"


using namespace cv;
using namespace cv::gpu;
using namespace rapidxml;
#define NUM_DICT 15
#define NUM_SUBJECTS 15
#define COLS 30
#define ROWS 400


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

template <class T>
bool contains(const std::vector<T> &vec, const T &value)
{
	return std::find(vec.begin(), vec.end(), value) != vec.end();
}

void
mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
    int        i;
       
    /* Examine input (right-hand-side) arguments. */
    mexPrintf("\nThere are %d right-hand-side argument(s).", nrhs);
    for (i=0; i<nrhs; i++)  {
        mexPrintf("\n\tInput Arg %i is of type:\t%s ",i,mxGetClassName(prhs[i]));
    }
    
    /* Examine output (left-hand-side) arguments. */
    mexPrintf("\n\nThere are %d left-hand-side argument(s).\n", nlhs);
    if (nlhs > nrhs)
      mexErrMsgIdAndTxt( "MATLAB:mexfunction:inputOutputMismatch",
              "Cannot specify more outputs than inputs.\n");
    
    for (i=0; i<nlhs; i++)  {
        plhs[i]=mxCreateDoubleMatrix(1,1,mxREAL);
        *mxGetPr(plhs[i])=(double)mxGetNumberOfElements(prhs[i]);
    }
}



int main(int argc, const char *argv[])
{
	/*
	xml_document<> doc;
	std::ifstream file("utdsigsets//face_walking_video_0.xml");
	std::stringstream buffer;
	buffer << file.rdbuf();
	file.close();
	std::string content(buffer.str());
	doc.parse<0>(&content[0]);

	xml_node<> *pRoot = doc.first_node();
	int count = 0;
	for(xml_node<> *pNode=pRoot->first_node("biometric-signature"); pNode; pNode=pNode->next_sibling())
	{

	count++;

	}
	std::cout << count << std::endl;
	*/



	cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());

	string cascadeName;
	string inputName = "C:\\Datasets\\UTD\\04802v125.dv";
	bool isInputImage = false;
	bool isInputVideo = true;
	bool isInputCamera = false;
	cascadeName = "HaarCascades\\haarcascade_frontalface_alt.xml";
	CascadeClassifier_GPU cascade_gpu;
	CascadeClassifier cascade_cpu;
	cascade_gpu.load(cascadeName);



	namedWindow("result", 1);

	Mat frame, frame_cpu, gray_cpu, resized_cpu, faces_downloaded, frameDisp;
	vector<Rect> facesBuf_cpu;

	GpuMat frame_gpu, gray_gpu, resized_gpu, facesBuf_gpu;


	bool useGPU = true;
	double scaleFactor = .5;
	bool findLargestObject = false;
	bool filterRects = false;
	bool helpScreen = false;
	Groupings g;
	int detections_num;


	bool flag = false;
	int var = 0;
	bool isSpaceBarPressed = false;
	bool makeInitialSegmentsFlag = false;

	double ** sim_matrix;

	sim_matrix = new double*[NUM_SUBJECTS];
	for(int i = 0; i<NUM_SUBJECTS; i++){
		sim_matrix[i] = new double[NUM_SUBJECTS];
	}

	Mat faceROI;
	Mat faceROI_resize;


	DIR *dir;
	struct dirent *ent;
	string name;
	string prev = "";
	int ID = 0;
	vector<string> names1;
	if ((dir = opendir ("Subjects\\")) != NULL) {

		while ((ent = readdir (dir)) != NULL) {
			name = ent->d_name;

			if(name.compare(".") != 0 && name.compare("..") != 0) {

				names1.push_back(name.substr((int)name.find('-')+1, 5));
				std::cout << name << std::endl;
				ID++;
			}
		}
	}
	std::cout << "Train or recog?: ";
	ID = 0;
	int choice;
	cin >> choice;

	std::cout << std::endl;
#pragma region TRAIN
	if(choice == 0){

		if ((dir = opendir ("C:\\Datasets\\UTD")) != NULL) {

			while ((ent = readdir (dir)) != NULL) {
				name = ent->d_name;

				if(name.compare(".") != 0 && name.compare("..") != 0) {
					if(name.substr(0, 5).compare(prev) != 0){
						ID++;
					}else{
						continue;
					}
					int numtimes = 0;
					Mat data;
					Mat temp;
					int numSegments = 2;

					VideoCapture capture;
					Mat image;
					capture.open("C:\\Datasets\\UTD\\"+name);
					if (!capture.isOpened())  // if not success, exit program
					{
						std::cout << "\nTrying Again....\n" << std::endl;
						capture.open(inputName);
					}

					double scale = (double)320/capture.get(CV_CAP_PROP_FRAME_WIDTH);
					for (;;)
					{
						if (isInputCamera || isInputVideo)
						{
							capture >> frame;

							if (frame.empty())
							{
								if(numtimes > 11){
									g = reset(data, g, numSegments);
									mxArray *args[3];

									int maxnum = 0;
									for(int j = 0; j<numSegments; j++){

										if(g.segments[j].size()>maxnum){
											maxnum = g.segments[j].size();
										}
									}
									args [0]= mxCreateDoubleMatrix(numSegments, maxnum, mxREAL);

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

									memcpy((void *)(mxGetPr(args[0])), data2, sizeof(data2)*maxnum*numSegments);

									args[1] = mxCreateDoubleMatrix(ROWS, numtimes, mxREAL);

									double * data1;
									data1 = new double[numtimes*ROWS];
									count = 0;
									for(int i = 0; i<numtimes; i++){
										for(int j = 0; j<400; j++){
											data1[count] = data.at<double>(j, i);
											count++;
										}

									}
									memcpy((void *)(mxGetPr(args[1])), data1, sizeof(data1)*400*numtimes);

									args[2] = mxCreateDoubleMatrix(1, 3, mxREAL);
									double * data3;
									data3 = new double[3];
									//numSegments, dict_size, internum
									
									data3[0] = 3;
									data3[1] = 3;
									data3[2] = 10;
									memcpy((void *)(mxGetPr(args[2])), data3, sizeof(data3)*3);

									mxArray * dict[2];
									dict[0] = mxCreateDoubleMatrix(ROWS, COLS, mxREAL);
									dict[1] = mxCreateDoubleMatrix(COLS, ROWS, mxREAL);

									mexCallMATLAB(2, dict, 3, args, "construct_dict.m");
									
									double ** temp_output = new double*[ROWS];
									for(int i = 0; i<ROWS; i++){
										temp_output[i] = new double[COLS];
									}

									memcpy(temp_output, (void *)(mxGetPr(dict[0])), sizeof(temp_output)*ROWS*COLS);

									for(int i = 0; i<ROWS; i++){
										for(int j = 0; j<COLS; j++){
											std::cout << setprecision(5) <<  temp_output[i][j] << " ";
										}
										std::cout << std::endl;
									}

									delete temp_output;
									delete data1;
									delete data2;
									delete data3;
								}
								break;
							}

						}

						(image.empty() ? frame : image).copyTo(frame_cpu);
						
						frame_gpu.upload(image.empty() ? frame : image);

						convertAndResize(frame_gpu, gray_gpu, resized_gpu, scale);


						TickMeter tm;
						tm.start();

						if (useGPU)
						{
							//cascade_gpu.visualizeInPlace = true;
							cascade_gpu.findLargestObject = findLargestObject;

							detections_num = cascade_gpu.detectMultiScale(resized_gpu, facesBuf_gpu,
								1.1, 2, Size(10, 10));
							facesBuf_gpu.colRange(0, detections_num).download(faces_downloaded);
						}

						if (useGPU)
						{
							resized_gpu.download(resized_cpu);

							for (int i = 0; i < detections_num; ++i)
							{
								Point c1;
								c1.x = (faces_downloaded.ptr<cv::Rect>()[i].x)*(1/scale);
								c1.y = (faces_downloaded.ptr<cv::Rect>()[i].y)*(1/scale);

								Point c2;

								c2.x = (faces_downloaded.ptr<cv::Rect>()[i].x+faces_downloaded.ptr<cv::Rect>()[i].width)*(1/scale);
								c2.y = (faces_downloaded.ptr<cv::Rect>()[i].y+faces_downloaded.ptr<cv::Rect>()[i].height)*(1/scale);
								if((faces_downloaded.ptr<cv::Rect>()[i].width)*(1/scale) < 100){
									rectangle(frame_cpu, c1, c2, Scalar(255), 1, 8, 0);
									faceROI = resized_cpu( faces_downloaded.ptr<cv::Rect>()[i] );
									flag = true;
								}
							}


						}


						if(flag){

							resize(faceROI, faceROI_resize, Size(20, 20), 0, 0, 1);
							if(numtimes == 0){
								faceROI_resize.reshape(1, 400).convertTo(data, CV_64F, 1, 0);
								data = data/255.0;
							}else{
								faceROI_resize.reshape(1, 400).convertTo(temp, CV_64F, 1, 0);
								temp = temp/255.0;

								hconcat(data, temp, data);
							}
							numtimes++;


							makeInitialSegmentsFlag = false;

							if(numtimes == 10){
								numSegments++;
								g = seg(data, numSegments);

							}


							if(numtimes > 10 && (numtimes)%4 == 0){
								g = addFrame(g, data.colRange(Range(numtimes-4, numtimes)), numtimes, numSegments);
							}

							flag = false;



						}
						tm.stop();
						double detectionTime = tm.getTimeMilli();
						double fps = 1000 / detectionTime;
						imshow("result", frame_cpu);

						if(waitKey(5) == 27)
							return 0;
					}

				}
				prev = name.substr(0, 5);
			}
			closedir (dir);
		} else {

			perror ("");
			return EXIT_FAILURE;
		}

#pragma endregion
	}else if(choice == 1){

		ofstream fout;
		fout.open("matrix.txt");



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
		string *names = new string[NUM_SUBJECTS];

		ID = 0;


		if ((dir = opendir ("C:\\Datasets\\UTD")) != NULL) {

			while ((ent = readdir (dir)) != NULL && ID<NUM_SUBJECTS) {
				name = ent->d_name;
				int numtimes = 0;
				if(name.compare(".") != 0 && name.compare("..") != 0) {
					if(name.substr(6, 1).compare("1") != 0 && contains(names1, name.substr(0, 5))){
						ID++;
					}else{
						continue;
					}
					Mat data;
					Mat temp;
					int numSegments = 2;

					VideoCapture capture;
					Mat image;
					capture.open("C:\\Datasets\\UTD\\"+name);
					if (!capture.isOpened())  // if not success, exit program
					{
						std::cout << "\nTrying Again....\n" << std::endl;
						capture.open(inputName);
					}
					double scale = (double)320/capture.get(CV_CAP_PROP_FRAME_WIDTH);

					for (;;)
					{
						if (isInputCamera || isInputVideo)
						{
							capture >> frame;
							if (frame.empty())
							{	

								vector<int> scores;
								scores.reserve(NUM_SUBJECTS);
								int max = 0;
								int id = 0;
								int max1 = 0;
								int id1 = 0;
								for(int i = 0; i<NUM_SUBJECTS; i++){
									scores.push_back(0);
								}
								std::cout << name << std::endl;

								sim_matrix[ID-1] = image_test(data, dict, pinvD);
								for(int z = 0; z<NUM_SUBJECTS; z++){
									fout << names1.at(z) << " " << name << " " << z << " " << ID-1 << " " << (z==ID-1 ? 1 : 0) << " " << sim_matrix[ID-1][z] << std::endl;
								}
								std::cout << ID-1 << std::endl;

								break;
							}	
						}

						(image.empty() ? frame : image).copyTo(frame_cpu);

						frame_gpu.upload(image.empty() ? frame : image);

						convertAndResize(frame_gpu, gray_gpu, resized_gpu, scale);


						TickMeter tm;
						tm.start();

						if (useGPU)
						{
							//cascade_gpu.visualizeInPlace = true;
							cascade_gpu.findLargestObject = findLargestObject;

							detections_num = cascade_gpu.detectMultiScale(resized_gpu, facesBuf_gpu,
								1.1, 2, Size(10, 10));
							facesBuf_gpu.colRange(0, detections_num).download(faces_downloaded);
						}

						if (useGPU)
						{
							resized_gpu.download(resized_cpu);

							for (int i = 0; i < detections_num; ++i)
							{
								Point c1;
								c1.x = (faces_downloaded.ptr<cv::Rect>()[i].x)*(1/scale);
								c1.y = (faces_downloaded.ptr<cv::Rect>()[i].y)*(1/scale);

								Point c2;
								c2.x = (faces_downloaded.ptr<cv::Rect>()[i].x+faces_downloaded.ptr<cv::Rect>()[i].width)*(1/scale);
								c2.y = (faces_downloaded.ptr<cv::Rect>()[i].y+faces_downloaded.ptr<cv::Rect>()[i].height)*(1/scale);
								if((faces_downloaded.ptr<cv::Rect>()[i].width)*(1/scale) < 100){
									rectangle(frame_cpu, c1, c2, Scalar(255), 1, 8, 0);
									faceROI = resized_cpu( faces_downloaded.ptr<cv::Rect>()[i] );
									flag = true;
								}
							}


						}


						if(flag){

							resize(faceROI, faceROI_resize, Size(20, 20), 0, 0, 1);
							equalizeHist(faceROI_resize, faceROI_resize);
							if(numtimes == 0){
								faceROI_resize.reshape(1, 400).convertTo(data, CV_64F, 1, 0);
								data = data/255.0;
							}else{
								faceROI_resize.reshape(1, 400).convertTo(temp, CV_64F, 1, 0);
								temp = temp/255.0;

								hconcat(data, temp, data);
							}
							numtimes++;


							makeInitialSegmentsFlag = false;

							if(numtimes == 10){
								numSegments++;
								g = seg(data, numSegments);

							}


							if(numtimes > 10 && (numtimes)%4 == 0){
								g = addFrame(g, data.colRange(Range(numtimes-4, numtimes)), numtimes, numSegments);
							}

							flag = false;



						}
						tm.stop();
						double detectionTime = tm.getTimeMilli();
						double fps = 1000 / detectionTime;
						

						if(waitKey(5) == 27)
							return 0;

					}
					prev = name.substr(0, 5);
				}

			}
		}
		fout.close();
	}

	ofstream fout;
	fout.open("matrix1.txt");
	for(int i = 0; i<NUM_SUBJECTS; i++){
		for(int j = 0; j<NUM_SUBJECTS; j++){
			fout << setprecision(6) << sim_matrix[i][j] << " ";
		}
		fout << std::endl;
	}
	fout.close();


	return 0;

}
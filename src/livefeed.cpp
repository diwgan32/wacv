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
using namespace std;
using namespace cv;
using namespace cv::gpu;
#define NUM_DICT 3
#define NUM_SUBJECTS 3
#define COLS 3
#define ROWS 400


static void help()
{
	cout << "Usage: ./cascadeclassifier_gpu \n\t--cascade <cascade_file>\n\t(<image>|--video <video>|--camera <camera_id>)\n"
		"Using OpenCV version " << CV_VERSION << endl << endl;
}


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


	cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());

	string cascadeName;
	string inputName = "C:\\Datasets\\UTD\\04802v125.dv";
	bool isInputImage = false;
	bool isInputVideo = true;
	bool isInputCamera = false;
	cascadeName = "HaarCascades\\haarcascade_frontalface_alt.xml";
	CascadeClassifier_GPU cascade_gpu;
	CascadeClassifier cascade_cpu;
	if (!cascade_gpu.load(cascadeName))
	{
		return cerr << "ERROR: Could not load cascade classifier \"" << cascadeName << "\"" << endl, help(), -1;
	}



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



	Mat faceROI;
	Mat faceROI_resize;


	DIR *dir;
	struct dirent *ent;
	string name;
	string prev = "";
	int ID = 0;
	if ((dir = opendir ("C:\\Datasets\\UTD")) != NULL) {
		/* print all the files and directories within directory */
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
					cout << "\nTrying Again....\n" << endl;
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
							g = reset(data, g, numSegments);
							MATFile *pmat;
							mxArray *pa1;

							int maxnum = 0;
							for(int j = 0; j<numSegments; j++){

								if(g.segments[j].size()>maxnum){
									maxnum = g.segments[j].size();
								}
							}
							pa1 = mxCreateDoubleMatrix(numSegments, maxnum, mxREAL);
							string filename = "Segments\\"+itos(ID)+".mat";
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



							pa1 = mxCreateDoubleMatrix(400, numtimes, mxREAL);
							filename = "Subjects\\"+itos(ID)+".mat";
							pmat = matOpen(filename.c_str(), "w");

							double * data1;
							data1 = new double[numtimes*400];
							count = 0;
							for(int i = 0; i<numtimes; i++){
								for(int j = 0; j<400; j++){
									data1[count] = data.at<double>(j, i);
									count++;
								}

							}


							memcpy((void *)(mxGetPr(pa1)), data1, sizeof(data1)*400*numtimes);
							status = matPutVariable(pmat, "SubjectData", pa1);
							if (status != 0) {

								printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
								return(EXIT_FAILURE);
							}  

							mxDestroyArray(pa1);
							matClose(pmat);

							delete data1;
							delete data2;
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
							rectangle(frame_cpu, c1, c2, Scalar(255), 1, 8, 0);
							faceROI = resized_cpu( faces_downloaded.ptr<cv::Rect>()[i] );
							flag = true;
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

					char key = waitKey(5);
				}

			}
			prev = name.substr(0, 5);
		}
		closedir (dir);
	} else {
		/* could not open directory */
		perror ("");
		return EXIT_FAILURE;
	}


	if (getCudaEnabledDeviceCount() == 0)
	{
		int x;
		cin >> x;
		return cerr << "No GPU found or the library is compiled without GPU support" << endl, -1;
	}




	return 0;

}



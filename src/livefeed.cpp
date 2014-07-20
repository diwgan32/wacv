// WARNING: this sample is under construction! Use it on your own risk.
#if defined _MSC_VER && _MSC_VER >= 1400
#pragma warning(disable : 4100)
#endif

#include "includes.h"
#include "Group.h"
#include "mp_test.h"
#include "segment.h"
#include "utils.h"
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

	if (getCudaEnabledDeviceCount() == 0)
	{
		int x;
		cin >> x;
		return cerr << "No GPU found or the library is compiled without GPU support" << endl, -1;
	}

	cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());

	string cascadeName;
	string inputName = "C:\\Users\\diwakar\\Downloads\\VideoFeed\\PaSCSamples\\02463d3328.mp4";
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


	VideoCapture capture;
	Mat image;
	capture.open(inputName);
	if (!capture.isOpened())  // if not success, exit program
	{
		cout << "\nTrying Again....\n" << endl;
		capture.open(inputName);
	}

	double scale = (double)320/capture.get(CV_CAP_PROP_FRAME_WIDTH);
	namedWindow("result", 1);

	Mat frame, frame_cpu, gray_cpu, resized_cpu, faces_downloaded, frameDisp;
	vector<Rect> facesBuf_cpu;

	GpuMat frame_gpu, gray_gpu, resized_gpu, facesBuf_gpu;

	/* parameters */
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

	int numtimes = 0;
	Mat data;
	Mat temp;
	int numSegments = 2;

	Mat faceROI;
	Mat faceROI_resize;

	for (;;)
	{
		if (isInputCamera || isInputVideo)
		{
			capture >> frame;
			if (frame.empty())
			{
				g = reset(data, g, numSegments);
				std::ofstream ifs("data.txt");

				for(int i = 0; i<numSegments; i++){
					ifs << "bestsgt{" << i+1 <<  "} = [";
					for(int j = 0; j<g.segments[i].size(); j++){
						ifs <<g.segments[i].at(j)+1 << " ";
					}
					ifs << "];";
					ifs << endl;
				}

				ifs.close();


				MATFile *pmat;
				mxArray *pa1;

				pa1 = mxCreateDoubleMatrix(400, numtimes, mxREAL);
				string filename = "Subjects\\"+itos(7)+".mat";
				pmat = matOpen(filename.c_str(), "w");

				double * data1;
				data1 = new double[numtimes*400];
				int count = 0;
				for(int i = 0; i<numtimes; i++){
					for(int j = 0; j<400; j++){
						data1[count] = data.at<double>(j, i);

						count++;
					}
				}

				memcpy((void *)(mxGetPr(pa1)), data1, sizeof(data1)*400*numtimes);
				int status = matPutVariable(pmat, "ImgData1", pa1);
				if (status != 0) {

					printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
					return(EXIT_FAILURE);
				}  

				mxDestroyArray(pa1);
				matClose(pmat);
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
				rectangle(resized_cpu, faces_downloaded.ptr<cv::Rect>()[i], Scalar(255));
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
		imshow("result", resized_cpu);

		char key = waitKey(5);
	}

	return 0;
}

// WARNING: this sample is under construction! Use it on your own risk.
#if defined _MSC_VER && _MSC_VER >= 1400
#pragma warning(disable : 4100)
#endif


#include <iostream>
#include <iomanip>
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;


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


static void matPrint(Mat &img, int lineOffsY, Scalar fontColor, const string &ss)
{
    int fontFace = FONT_HERSHEY_DUPLEX;
    double fontScale = 0.8;
    int fontThickness = 2;
    Size fontSize = cv::getTextSize("T[]", fontFace, fontScale, fontThickness, 0);

    Point org;
    org.x = 1;
    org.y = 3 * fontSize.height * (lineOffsY + 1) / 2;
    putText(img, ss, org, fontFace, fontScale, CV_RGB(0,0,0), 5*fontThickness/2, 16);
    putText(img, ss, org, fontFace, fontScale, fontColor, fontThickness, 16);
}


static void displayState(Mat &canvas, bool bHelp, bool bGpu, bool bLargestFace, bool bFilter, double fps)
{
    Scalar fontColorRed = CV_RGB(255,0,0);
    Scalar fontColorNV  = CV_RGB(118,185,0);

    ostringstream ss;
    ss << "FPS = " << setprecision(1) << fixed << fps;
    matPrint(canvas, 0, fontColorRed, ss.str());
    ss.str("");
    ss << "[" << canvas.cols << "x" << canvas.rows << "], " <<
        (bGpu ? "GPU, " : "CPU, ") <<
        (bLargestFace ? "OneFace, " : "MultiFace, ") <<
        (bFilter ? "Filter:ON" : "Filter:OFF");
    matPrint(canvas, 1, fontColorRed, ss.str());

    // by Anatoly. MacOS fix. ostringstream(const string&) is a private
    // matPrint(canvas, 2, fontColorNV, ostringstream("Space - switch GPU / CPU"));
    if (bHelp)
    {
        matPrint(canvas, 2, fontColorNV, "Space - switch GPU / CPU");
        matPrint(canvas, 3, fontColorNV, "M - switch OneFace / MultiFace");
        matPrint(canvas, 4, fontColorNV, "F - toggle rectangles Filter");
        matPrint(canvas, 5, fontColorNV, "H - toggle hotkeys help");
        matPrint(canvas, 6, fontColorNV, "1/Q - increase/decrease scale");
    }
    else
    {
        matPrint(canvas, 2, fontColorNV, "H - toggle hotkeys help");
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
    string inputName = "../../../../PaSCSamples/02463d3666.mp4";
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

    int detections_num;
    for (;;)
    {
        if (isInputCamera || isInputVideo)
        {
            capture >> frame;
            if (frame.empty())
            {
                break;
            }
        }

        (image.empty() ? frame : image).copyTo(frame_cpu);
        frame_gpu.upload(image.empty() ? frame : image);

        convertAndResize(frame_gpu, gray_gpu, resized_gpu, scaleFactor);


        TickMeter tm;
        tm.start();

        if (useGPU)
        {
            //cascade_gpu.visualizeInPlace = true;
            cascade_gpu.findLargestObject = findLargestObject;

            detections_num = cascade_gpu.detectMultiScale(resized_gpu, facesBuf_gpu,
                                                         1.1, 2, Size(50, 50));
            facesBuf_gpu.colRange(0, detections_num).download(faces_downloaded);
        }
        
        if (useGPU)
        {
            resized_gpu.download(resized_cpu);

             for (int i = 0; i < detections_num; ++i)
             {
                rectangle(resized_cpu, faces_downloaded.ptr<cv::Rect>()[i], Scalar(255));
             }
        }

        tm.stop();
        double detectionTime = tm.getTimeMilli();
        double fps = 1000 / detectionTime;

        //print detections to console
        cout << setfill(' ') << setprecision(2);
        cout << setw(6) << fixed << fps << " FPS, " << detections_num << " det";
        if ((filterRects || findLargestObject) && detections_num > 0)
        {
            Rect *faceRects = useGPU ? faces_downloaded.ptr<Rect>() : &facesBuf_cpu[0];
            for (int i = 0; i < min(detections_num, 2); ++i)
            {
                cout << ", [" << setw(4) << faceRects[i].x
                     << ", " << setw(4) << faceRects[i].y
                     << ", " << setw(4) << faceRects[i].width
                     << ", " << setw(4) << faceRects[i].height << "]";
            }
        }
        cout << endl;

        cvtColor(resized_cpu, frameDisp, CV_GRAY2BGR);
        displayState(frameDisp, helpScreen, useGPU, findLargestObject, filterRects, fps);
        imshow("result", frameDisp);

        char key = (char)waitKey(5);
        if (key == 27)
        {
            break;
        }

        switch (key)
        {
        case ' ':
            useGPU = !useGPU;
            break;
        case 'm':
        case 'M':
            findLargestObject = !findLargestObject;
            break;
        case 'f':
        case 'F':
            filterRects = !filterRects;
            break;
        case '1':
            scaleFactor *= 1.05;
            break;
        case 'q':
        case 'Q':
            scaleFactor /= 1.05;
            break;
        case 'h':
        case 'H':
            helpScreen = !helpScreen;
            break;
        }
    }

    return 0;
}

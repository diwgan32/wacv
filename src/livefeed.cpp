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
#define COLS 15
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



int main(int argc, const char *argv[])
{


	cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice()); //Load GPU

	/* Load cascades */
	string cascadeName = "HaarCascades\\haarcascade_frontalface_alt.xml";
	CascadeClassifier_GPU cascade_gpu;
	CascadeClassifier cascade_cpu;
	cascade_gpu.load(cascadeName);

	Mat d;
	Mat c;

	Mat f = c*d;

	/*Initialize matrices*/
	Mat frame, frame_cpu, gray_cpu, resized_cpu, faces_downloaded, faceROI, faceROI_resize;
	vector<Rect> facesBuf_cpu;
	GpuMat frame_gpu, gray_gpu, resized_gpu, facesBuf_gpu;

	int detections_num;


	bool flag = false;

	bool isSpaceBarPressed = false;
	bool makeInitialSegmentsFlag = false;

	std::cout << "Train or recog?: ";

	int choice;
	cin >> choice;

	if(choice == 3){
		DIR *dir;
		struct dirent *ent;
		string name;
		vector<string> target_names;

		int target_count = 0;
		if ((dir = opendir ("Subjects1\\")) != NULL) {

			while ((ent = readdir (dir)) != NULL) {
				name = ent->d_name;
				if(name.compare(".") != 0 && name.compare("..") != 0 ) {
					target_names.push_back(name);

				}
			}
		}
		int ID = 0;
		while(ID<target_names.size()){
			name = target_names.at(ID);
			MATFile *pMat;
			pMat = matOpen(("Subjects1\\"+target_names.at(ID)).c_str(), "r");
			mxArray * subjectData;
			subjectData = matGetVariable(pMat, "SubjectData");

			int numFrames = mxGetN(subjectData);
			double * data1 = new double[400*numFrames];
			memcpy(data1, (void *)(mxGetPr(subjectData)), sizeof(data1)*400*numFrames);

			int count = 0;
			Mat data(400, numFrames, CV_64F);
			for(int i = 0; i<numFrames; i++){
				for(int j = 0; j<400; j++){
					data.at<double>(j, i) = data1[count];
					count++;
				}
			}

			matClose(pMat);
			delete data1;
			mxDestroyArray(subjectData);

			int numSegments = 4;
			Groupings g;
			g = seg(data.colRange(Range(0, 10)), numSegments);

			int numberOfFramesCollected = 10;

			while(numberOfFramesCollected+4 < numFrames){
				numberOfFramesCollected += 4;
				g = addFrame(g, data.colRange(Range(numberOfFramesCollected-4, numberOfFramesCollected)), numberOfFramesCollected, numSegments);
			}

			g = reset(data, g, numSegments);

			int maxnum = 0;
			for(int j = 0; j<numSegments; j++){

				if(g.segments[j].size()>maxnum){
					maxnum = g.segments[j].size();
				}
			}
			mxArray * pa1 = mxCreateDoubleMatrix(numSegments, maxnum, mxREAL);
			string filename = "Segments\\"+name;
			pMat  = matOpen(filename.c_str(), "w");

			/* 
			* Since segment data is not an exact rectangular
			* array, all the holes in the data are filled
			* with -1s.
			*/
			double * data2;
			count = 0;
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
			int status = matPutVariable(pMat, "Segments", pa1);
			if (status != 0) {

				printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
				return(EXIT_FAILURE);
			}  

			delete data2;
			mxDestroyArray(pa1);
			matClose(pMat);
			ID++;
		}
	}

	if(choice == 0){
		DIR *dir;
		struct dirent *ent;
		string name;
		vector<string> target_names;

		int target_count = 0;
		if ((dir = opendir ("Subjects\\")) != NULL) {

			while ((ent = readdir (dir)) != NULL) {
				name = ent->d_name;
				if(name.compare(".") != 0 && name.compare("..") != 0 ) {
					target_names.push_back(name);

				}
			}
		}
		int ID = 0;


		/* Ask user to input which video in UTD dataset to start training on */
		string startPos;
		cout << "Enter starting video, or 0 to start at beginning: ";
		cin >> startPos;
		cout << endl;

		/* Offset the ID variable accordingly */
		if(startPos.compare("0") == 0){
			ID = 0;
		}else{
			while(target_names.at(ID).compare(startPos) != 0){
				ID++;
			}
		}

		Groupings g;
		//Loop through target names
		while(ID<target_names.size()){
			cout << ID << "   ";
			string name = target_names.at(ID);

			int numberOfFramesCollected = 0; 

			Mat setOfAllFramesCollected;
			Mat temp;
			int numSegments = 2;

			VideoCapture capture;
			capture.open("C:\\UTD\\"+target_names.at(ID));
			double scale = (double)320/capture.get(CV_CAP_PROP_FRAME_WIDTH);



			for (;;)
			{
				capture >> frame; //Get next frame from stream
#pragma region WRITE_DATA
				if (frame.empty())
				{
					// Only write data if enough frames were collected
					if(numberOfFramesCollected > 10){

						g = reset(setOfAllFramesCollected, g, numSegments);
						MATFile *pmat;
						mxArray *pa1;

						int maxnum = 0;
						for(int j = 0; j<numSegments; j++){

							if(g.segments[j].size()>maxnum){
								maxnum = g.segments[j].size();
							}
						}
						pa1 = mxCreateDoubleMatrix(numSegments, maxnum, mxREAL);
						string filename = "Segments\\"+name;
						pmat = matOpen(filename.c_str(), "w");

						/* 
						* Since segment data is not an exact rectangular
						* array, all the holes in the data are filled
						* with -1s.
						*/
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



						pa1 = mxCreateDoubleMatrix(400, numberOfFramesCollected, mxREAL);
						filename = "Subjects\\"+itos(ID)+"-"+name.substr(0, 9)+".mat";
						pmat = matOpen(filename.c_str(), "w");

						double * data1;
						data1 = new double[numberOfFramesCollected*400];
						count = 0;
						for(int i = 0; i<numberOfFramesCollected; i++){
							for(int j = 0; j<400; j++){
								data1[count] = setOfAllFramesCollected.at<double>(j, i);
								count++;
							}

						}


						memcpy((void *)(mxGetPr(pa1)), data1, sizeof(data1)*400*numberOfFramesCollected);
						status = matPutVariable(pmat, "SubjectData", pa1);

						if (status != 0) {

							printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
							return(EXIT_FAILURE);
						}  

						mxDestroyArray(pa1);
						matClose(pmat);

						delete data1;
						delete data2;
					}
					break;
				}
#pragma endregion
				TickMeter tm;
				tm.start();
#pragma region ACCUMULATE_FRAMES
				frame.copyTo(frame_cpu);
				frame_gpu.upload(frame);

				convertAndResize(frame_gpu, gray_gpu, resized_gpu, scale);


				cascade_gpu.findLargestObject = true;

				// Find faces
				detections_num = cascade_gpu.detectMultiScale(resized_gpu, facesBuf_gpu,
					1.1, 2, Size(10, 10));
				facesBuf_gpu.colRange(0, detections_num).download(faces_downloaded);
				resized_gpu.download(resized_cpu);

				/*
				Loop through every single face. Important to note
				that only the last face on the list will be
				taken into account
				*/
				for (int i = 0; i < detections_num; ++i)
				{
					Point c1;
					c1.x = (faces_downloaded.ptr<cv::Rect>()[i].x)*(1/scale);
					c1.y = (faces_downloaded.ptr<cv::Rect>()[i].y)*(1/scale);

					Point c2;

					c2.x = (faces_downloaded.ptr<cv::Rect>()[i].x+faces_downloaded.ptr<cv::Rect>()[i].width)*(1/scale);
					c2.y = (faces_downloaded.ptr<cv::Rect>()[i].y+faces_downloaded.ptr<cv::Rect>()[i].height)*(1/scale);
					if((faces_downloaded.ptr<cv::Rect>()[i].width)*(1/scale) < 100 && 

						faces_downloaded.ptr<cv::Rect>()[i].y*(1/scale) < 250){


							rectangle(frame_cpu, c1, c2, Scalar(255), 1, 8, 0);
							faceROI = resized_cpu( faces_downloaded.ptr<cv::Rect>()[i] );
							flag = true;
					}
				}

				// Only add faces to the set if a face was detected
				if(flag){

					resize(faceROI, faceROI_resize, Size(20, 20), 0, 0, 1);
					equalizeHist(faceROI_resize, faceROI_resize);

					/* 
					If first frame, add to set, otherwise, use temp matrix and 
					concactenate matrices
					*/

					if(numberOfFramesCollected == 0){
						faceROI_resize.reshape(1, 400).convertTo(setOfAllFramesCollected, CV_64F, 1, 0);
						setOfAllFramesCollected = setOfAllFramesCollected/255.0;
					}else{
						faceROI_resize.reshape(1, 400).convertTo(temp, CV_64F, 1, 0);
						temp = temp/255.0;
						hconcat(setOfAllFramesCollected, temp, setOfAllFramesCollected);
					}
					numberOfFramesCollected++;


					makeInitialSegmentsFlag = false;

					// If 10 frames have been collected, create segments
					if(numberOfFramesCollected == 10){
						g = seg(setOfAllFramesCollected, numSegments);

					}

					//Keep adding frames to set every time 4 frames have been accumulated
					if(numberOfFramesCollected > 10 && (numberOfFramesCollected)%4 == 0){
						g = addFrame(g, setOfAllFramesCollected.colRange(Range(numberOfFramesCollected-4, numberOfFramesCollected)), numberOfFramesCollected, numSegments);
					}

					flag = false;



				}
#pragma endregion

				/* Performance metrics */
				tm.stop();
				double detectionTime = tm.getTimeMilli();
				double fps = 1000 / detectionTime;
				imshow("result", frame_cpu);

				if(waitKey(5) == 27)
					return 0;
			}

			ID++;

		}
	}else if (choice == 1){
		Groupings g;
		ofstream fout;
		fout.open("SimMat.txt");
		ofstream fout1;
		fout1.open("SimMat1.txt");

		int numTargets = 0;
		DIR *dir;
		struct dirent *ent;
		string name;
		vector<string> target_names;

		int target_count = 0;

		if ((dir = opendir ("Subjects\\")) != NULL) {

			while ((ent = readdir (dir)) != NULL) {
				name = ent->d_name;
				if(name.compare(".") != 0 && name.compare("..") != 0 ) {
					target_names.push_back(name.substr(0, 9));

				}
			}
		}

		vector<string> query_names;	
		xml_document<> doc;
		std::ifstream file("utdsigsets//face_walking_video_0.xml");
		std::stringstream buffer;
		buffer << file.rdbuf();
		file.close();
		std::string content(buffer.str());
		doc.parse<0>(&content[0]);

		xml_node<> *pRoot = doc.first_node();

		string prev = "";


		for(xml_node<> *pNode=pRoot->first_node("biometric-signature"); pNode; pNode=pNode->next_sibling())
		{
			query_names.push_back(string(pNode->first_node("presentation")->first_attribute("file-name")->value()).substr(21, 9));
		}


		while(target_count < target_names.size()){
			cout << target_count << "\t";
			Mat D = readBin(("Dictionaries\\"+target_names.at(target_count)+".bin").c_str(), ROWS, COLS);
			Mat pinvD = readBin(("InverseDictionaries\\"+target_names.at(target_count)+".bin").c_str(), COLS, ROWS);

			int query_count = 0;

			while(query_count < query_names.size()){

				if(target_names.at(target_count).compare(query_names.at(query_count)) != 0){
					if(!contains(target_names, query_names.at(query_count))){
						query_count++;
						continue;
					}
					MATFile *pMat;
					pMat = matOpen(("Subjects\\"+query_names.at(query_count)+".mat").c_str(), "r");
					mxArray * subjectData;
					subjectData = matGetVariable(pMat, "SubjectData");

					int numFrames = mxGetN(subjectData);
					double * data1 = new double[400*numFrames];
					memcpy(data1, (void *)(mxGetPr(subjectData)), sizeof(data1)*400*numFrames);

					int count = 0;
					Mat data(400, numFrames, CV_64F);
					for(int i = 0; i<numFrames; i++){
						for(int j = 0; j<400; j++){
							data.at<double>(j, i) = data1[count];
							count++;
						}
					}

					matClose(pMat);
					delete data1;
					mxDestroyArray(subjectData);

					pMat = matOpen(("Segments\\"+query_names.at(query_count)+".mat").c_str(), "r");
					mxArray * segmentData;
					segmentData = matGetVariable(pMat, "Segments");

					int maxNumberOfFramesPerSegment = mxGetN(segmentData);
					double * data2 = new double[3*maxNumberOfFramesPerSegment];
					memcpy(data2, (void *)(mxGetPr(segmentData)), sizeof(data2)*3*maxNumberOfFramesPerSegment);

					matClose(pMat);
					mxDestroyArray(segmentData);

					Groupings g;
					g.segments = new vector<int>[3];
					count  = 0;
					for(int i = 0; i<maxNumberOfFramesPerSegment; i++){
						for(int j = 0; j<3; j++){
							if(data2[count] != -1){
								g.segments[j].push_back(data2[count]);
							}
							count++;
						}
					}

					double sim = image_test(data, D, pinvD);
					if(target_names.at(target_count).substr(0, 5).compare(query_names.at(query_count).substr(0, 5)) == 0){
						fout << "1 " << endl;
						fout1 << 20-sim << endl;
					}else{
						fout << "0 " << endl;
						fout1 << 20-sim << endl;
					}

				}
				query_count++;
			}
			target_count++;

		}

		fout1.close();
		fout.close();
	}

	return 0;

}
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


	/*Initialize matrices*/
	Mat frame, frame_cpu, gray_cpu, resized_cpu, faces_downloaded, faceROI, faceROI_resize;
	vector<Rect> facesBuf_cpu;
	GpuMat frame_gpu, gray_gpu, resized_gpu, facesBuf_gpu;

	int detections_num;


	bool flag = false;

	bool isSpaceBarPressed = false;
	bool makeInitialSegmentsFlag = false;

	cout << endl << endl;
	int choice = atoi(argv[1]);
	int ROWS = atoi(argv[2]);
	bool isHist = false;

	if(string(argv[3]).compare("-hist") == 0){
		isHist = true;
	}

	int numSegments = atoi(argv[4]);

	if(choice == 3){
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
		while(ID<target_names.size()){
			name = target_names.at(ID);
			MATFile *pMat;
			pMat = matOpen(("Subjects\\"+target_names.at(ID)).c_str(), "r");
			mxArray * subjectData;
			subjectData = matGetVariable(pMat, "SubjectData");

			int numFrames = mxGetN(subjectData);
			double * data1 = new double[ROWS*numFrames];
			memcpy(data1, (void *)(mxGetPr(subjectData)), sizeof(data1)*ROWS*numFrames);

			int count = 0;
			Mat data(ROWS, numFrames, CV_64F);
			for(int i = 0; i<numFrames; i++){
				for(int j = 0; j<ROWS; j++){
					data.at<double>(j, i) = data1[count];
					count++;
				}
			}

			matClose(pMat);

			delete data1;

			mxDestroyArray(subjectData);

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
		vector<string> target_names;	
		xml_document<> doc;

		/*Load sigsets*/
		std::ifstream file("utdsigsets\\face_walking_video.xml"); //Note: face_walking_video is UTD target sigset
		std::stringstream buffer;
		buffer << file.rdbuf();
		file.close();
		std::string content(buffer.str());
		doc.parse<0>(&content[0]);

		xml_node<> *pRoot = doc.first_node();
		int numberOfVideosToTrain = 0;
		int ID = 0;

		/*Loop through xml file and extract filenames on the target sigset*/
		for(xml_node<> *pNode=pRoot->first_node("biometric-signature"); pNode; pNode=pNode->next_sibling())
		{

			target_names.push_back(string(pNode->first_node("presentation")->first_attribute("file-name")->value()).substr(21));

			numberOfVideosToTrain++;
		}
		cout << "Number of videos to train: " << numberOfVideosToTrain << endl;






		//Loop through target names
		while(ID<target_names.size()){
			cout << ID << "   ";
			string name = target_names.at(ID).substr(0, 9);

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


						MATFile *pmat;
						mxArray *pa1;



						pa1 = mxCreateDoubleMatrix(ROWS, numberOfFramesCollected, mxREAL);
						string filename = "Subjects\\"+itos(ID)+"-"+name.substr(0, 9)+".mat";
						pmat = matOpen(filename.c_str(), "w");

						double * data1;
						data1 = new double[numberOfFramesCollected*ROWS];
						int count = 0;
						for(int i = 0; i<numberOfFramesCollected; i++){
							for(int j = 0; j<ROWS; j++){
								data1[count] = setOfAllFramesCollected.at<double>(j, i);
								count++;
							}

						}


						memcpy((void *)(mxGetPr(pa1)), data1, sizeof(data1)*ROWS*numberOfFramesCollected);
						int status = matPutVariable(pmat, "SubjectData", pa1);

						if (status != 0) {

							printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
							return(EXIT_FAILURE);
						}  

						mxDestroyArray(pa1);
						matClose(pmat);

						delete data1;
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

					resize(faceROI, faceROI_resize, Size(sqrt((double)ROWS), sqrt((double)ROWS)), 0, 0, 1);

					if(isHist){
						equalizeHist(faceROI_resize, faceROI_resize);

					}
					/* 
					If first frame, add to set, otherwise, use temp matrix and 
					concactenate matrices
					*/

					if(numberOfFramesCollected == 0){
						faceROI_resize.reshape(1, ROWS).convertTo(setOfAllFramesCollected, CV_64F, 1, 0);
						setOfAllFramesCollected = setOfAllFramesCollected/255.0;
					}else{
						faceROI_resize.reshape(1, ROWS).convertTo(temp, CV_64F, 1, 0);
						temp = temp/255.0;
						hconcat(setOfAllFramesCollected, temp, setOfAllFramesCollected);
					}
					numberOfFramesCollected++;


					makeInitialSegmentsFlag = false;
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
		int COLS = atoi(argv[5])*numSegments;
		cout << COLS << endl;
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

		xml_document<> doc1;
		std::ifstream file1("utdsigsets//face_walking_video_1.xml");
		std::stringstream buffer1;
		buffer1 << file1.rdbuf();
		file1.close();
		std::string content1(buffer1.str());
		doc1.parse<0>(&content1[0]);

		xml_node<> *pRoot1 = doc1.first_node();

		for(xml_node<> *pNode=pRoot1->first_node("biometric-signature"); pNode; pNode=pNode->next_sibling())
		{
			query_names.push_back(string(pNode->first_node("presentation")->first_attribute("file-name")->value()).substr(21, 9));
		}

		xml_document<> doc2;
		std::ifstream file2("utdsigsets//face_walking_video_2.xml");
		std::stringstream buffer2;
		buffer2 << file2.rdbuf();
		file2.close();
		std::string content2(buffer2.str());
		doc2.parse<0>(&content2[0]);

		xml_node<> *pRoot2 = doc2.first_node();

		for(xml_node<> *pNode=pRoot2->first_node("biometric-signature"); pNode; pNode=pNode->next_sibling())
		{
			query_names.push_back(string(pNode->first_node("presentation")->first_attribute("file-name")->value()).substr(21, 9));
		}
		xml_document<> doc3;
		std::ifstream file3("utdsigsets//face_walking_video_3.xml");
		std::stringstream buffer3;
		buffer3 << file3.rdbuf();
		file3.close();
		std::string content3(buffer3.str());
		doc3.parse<0>(&content3[0]);

		xml_node<> *pRoot3 = doc3.first_node();

		for(xml_node<> *pNode=pRoot3->first_node("biometric-signature"); pNode; pNode=pNode->next_sibling())
		{
			query_names.push_back(string(pNode->first_node("presentation")->first_attribute("file-name")->value()).substr(21, 9));
		}
		double elapsed_secs = 0;
		ofstream times;
		while(target_count < target_names.size()){
			cout << target_count << "\t";
			Mat D = readBin(("Dictionaries\\"+target_names.at(target_count)+".bin").c_str(), ROWS, COLS);
			Mat pinvD = readBin(("InverseDictionaries\\"+target_names.at(target_count)+".bin").c_str(), COLS, ROWS);

			int query_count = 0;

			while(query_count < query_names.size()){

				if(target_names.at(target_count).compare(query_names.at(query_count)) != 0){

					MATFile *pMat;
					pMat = matOpen(("Subjects\\"+query_names.at(query_count)+".mat").c_str(), "r");
					if(pMat == NULL){
						if(target_names.at(target_count).substr(0, 5).compare(query_names.at(query_count).substr(0, 5)) == 0){
							fout << "1 " << endl;
							fout1 << -10 << " "<<endl;
						}else{
							fout << "0 " << endl;
							fout1 << -10 << " " <<endl;
						}
						query_count++;
					}else{
						mxArray * subjectData;
						subjectData = matGetVariable(pMat, "SubjectData");

						int numFrames = mxGetN(subjectData);
						double * data1 = new double[ROWS*numFrames];
						memcpy(data1, (void *)(mxGetPr(subjectData)), sizeof(data1)*ROWS*numFrames);

						int count = 0;
						Mat data(ROWS, numFrames, CV_64F);
						for(int i = 0; i<numFrames; i++){
							for(int j = 0; j<ROWS; j++){
								data.at<double>(j, i) = data1[count];
								count++;
							}
						}

						matClose(pMat);
						delete data1;
						mxDestroyArray(subjectData);


						clock_t begin = clock();
						double sim = image_test(data, D, pinvD);
						clock_t end = clock();
						elapsed_secs += double(end - begin) / CLOCKS_PER_SEC;

						
						if(target_names.at(target_count).substr(0, 5).compare(query_names.at(query_count).substr(0, 5)) == 0){
							fout << "1 " << endl;
							fout1 << 20-sim << endl;
						}else{
							fout << "0 " << endl;
							fout1 << 20-sim << endl;
						}
					}
				}
				query_count++;
			}
			target_count++;

		}
		times.open("times.txt");
		times << elapsed_secs << endl;
		times.close();
		fout1.close();
		fout.close();
	}

	return 0;

}

/*
int choice = atoi(argv[1]);
int ROWS = atoi(argv[2]);
bool isHist = false;

if(string(argv[3]).compare("-hist") == 0){
isHist = true;
}

int numSegments = atoi(argv[4]);
*/

/*


file.open("utdsigsets//face_walking_video_1.xml");
buffer << file.rdbuf();
file.close();
content = (buffer.str());
doc.parse<0>(&content[0]);

pRoot = doc.first_node();

for(xml_node<> *pNode=pRoot->first_node("biometric-signature"); pNode; pNode=pNode->next_sibling())
{
query_names.push_back(string(pNode->first_node("presentation")->first_attribute("file-name")->value()).substr(21, 9));
}

file.open("utdsigsets//face_walking_video_2.xml");
buffer << file.rdbuf();
file.close();
content = (buffer.str());
doc.parse<0>(&content[0]);

pRoot = doc.first_node();

for(xml_node<> *pNode=pRoot->first_node("biometric-signature"); pNode; pNode=pNode->next_sibling())
{
query_names.push_back(string(pNode->first_node("presentation")->first_attribute("file-name")->value()).substr(21, 9));
}
file.open("utdsigsets//face_walking_video_3.xml");
buffer << file.rdbuf();
file.close();
content = (buffer.str());
doc.parse<0>(&content[0]);

pRoot = doc.first_node();

for(xml_node<> *pNode=pRoot->first_node("biometric-signature"); pNode; pNode=pNode->next_sibling())
{
query_names.push_back(string(pNode->first_node("presentation")->first_attribute("file-name")->value()).substr(21, 9));
}



for(xml_node<> *pNode=pRoot->first_node("biometric-signature"); pNode; pNode=pNode->next_sibling())
{
query_names.push_back(string(pNode->first_node("presentation")->first_attribute("file-name")->value()).substr(21, 9));
}
*/
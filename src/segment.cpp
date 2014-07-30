#include "includes.h"
#include "Group.h"

using namespace cv;
using namespace std;


/* 
 * Function to determine initial groupings of frames
 * As frames are generated, addFrames() will be used 
 * to incorporate them into the groupings
*/

Groupings myKMeans(Mat Y, int k){

	Mat dist = Mat(Y.rows, Y.rows, CV_64F, 0.0f);

	double maxd = -100000.0;
	int imax = 0;
	int jmax = 0;

	for(int i = 0; i<Y.rows; i++){
		for(int j = i+1; j<Y.rows; j++){
			dist.at<double>(i, j) = norm(Y.row(i)-Y.row(j), 4);

			if(dist.at<double>(i, j)>maxd){
				maxd = dist.at<double>(i, j);
				imax = i;
				jmax = j;
			}
		}
	}

	vector<int> tbl (Y.rows);
	fill(tbl.begin(), tbl.end(), 1);

	vector<int> cand;
	cand.push_back(imax);
	cand.push_back(jmax);
	tbl.at(imax) = 0;
	tbl.at(jmax) = 0;

	double maxdist = -10000.0;
	double accdist = 0.0;

	int besti = 0;

	if(Y.rows >= k){
		while (cand.size() < k){
			for(int i=0; i<Y.rows; i++){

				if(tbl.at(i) == 0) continue;
				accdist = 1;

				for(int j = 0; j<cand.size(); j++){
					accdist = accdist * dist.at<double>(std::min(i, cand.at(j)), std::max(i, cand.at(j)));
				}

				if(accdist > maxdist){
					maxdist = accdist;
					besti = i;
				}

			}

			cand.push_back(besti);
			tbl.at(besti) = 0;
		}
	}


	Groupings g;
	g.segments = new vector<int>[k];
	for(int i = 0; i<k; i++){
		g.count.push_back(0);
	}
	int bestj = 0;
	double minD = 10000;
	double D = 0;
	Mat temp;
	for(int i = 0; i<Y.rows; i++){
		minD = INT_MAX;
		bestj = 0;
		for(int j = 0; j<k; j++){
			if(i == cand.at(j)){
				temp = Y.row(i);
				transpose(temp, temp);
				g.centers.push_back(temp);
				bestj = j;
				break;
			}
			D = norm(Y.row(i)-Y.row(cand.at(j)), 4);
			if(D<minD){
				minD = D;
				bestj = j;
			}
		}
		
		g.segments[bestj].push_back(i);
	
		g.count.at(bestj)+=1;
	}

	return g;
}

/*
 * Because the addFrames() function moves the centroids,
 * the groupings need to be redetermined as per the 
 * new centroids
*/
Groupings reset(Mat data, Groupings g, int k){
	double dist = 0;
	double mindist = 10000.0;
	int bestj = 0;
	for(int j = 0; j<k; j++){
		g.segments[j].clear();
		g.segments[j].reserve(data.cols/(k+1));
	}
	for(int i = 0; i<data.cols; i++){
		dist = 0;
		mindist = 10000.0;
		bestj = 0;
		for(int j = 0; j<k; j++){
			dist = norm(data.col(i)-g.centers.at(j), 4);
			//cout << dist << endl;
			if(dist<mindist){
				mindist = dist;
				bestj = j;
			}
		}
		g.segments[bestj].push_back(i);
	}
	return g;
}

/*
 * Add a set of 4 frames to the partitions as per the
 * mini-batch k-means algorithm
*/

Groupings addFrame(Groupings g, Mat frame, int frameid, int k){

	double dist = 0;
	double mindist = 10000.0;
	int bestj = 0;
	int centers[4];
	Mat temp = frame;
	//cout << g.centers.at(0) << endl;
	for(int i = 0; i<4; i++){
		dist = 0;
		mindist = DBL_MAX;
		bestj = 0;
		for(int j = 0; j<k; j++){
			dist = norm(frame.col(i)-g.centers.at(j), 4);

			if(dist<mindist){
				mindist = dist;
				bestj = j;
			}
		}
		g.segments[bestj].push_back(frameid-(3-i));
		centers[i] = bestj;

	}

	for(int i = 0; i<4; i++){
		Mat c = g.centers.at(centers[i]);

		g.count.at(centers[i]) = g.count.at(centers[i])+1;
		double rate = 1/(double)g.count.at(centers[i]);
		c = ((1-rate)*c)+(frame.col(i)*rate);
		g.centers.at(centers[i]) = c;
	}

	return g;



}

double scatter(Mat data, Groupings g, int k){
	double maxd = 0;
	double dist = 0;
	double max = 0;
	double scat = 0;
	Mat temp(400, 1, CV_64F);
	Mat temp1(1, 400, CV_64F, 0.0f);
	for(int i = 0; i<k; i++){

		for(int l = 0; l<g.segments[i].size(); l++){
			temp = data.col(g.segments[i].at(l))-g.centers.at(i);
			transpose(temp, temp);

			temp1 += ((data.col(g.segments[i].at(l))-g.centers.at(i))*temp);


		}
		cout << norm(temp1, 4) << " ";
		temp1 = 0;

	}
	cout << endl;
	return maxd;

}

/*
 * Determine the initial partitions
*/
Groupings seg(Mat ImgData, int k){

	Groupings g;
	g.segments = new vector<int>[k];
	for(int i = 0; i<k; i++){
		g.centers.push_back(Mat());
		g.count.push_back(0);
	}
	Mat Z = ImgData;
	transpose(Z, Z);

	flip(Z, Z, 0);

	int N = 1000;

	double maxscr = -10000;
	return myKMeans(Z, k);

}


/* 
 * Utility function that returns a set of matrices,
 * each one with the same pose and illumination.
*/

vector<Mat> splitBySegment(Mat data, Groupings g){
	size_t k = g.centers.size();
	vector<Mat> split;
	split.reserve(k);
	Mat temp;
	Mat temp1;
	for(int i = 0; i<k; i++){
		for(int j = 0; j<g.segments[i].size(); j++){
			if(j==0){
				temp = data.col(g.segments[i].at(j));
				split.push_back(temp);
			}else{
				temp = data.col(g.segments[i].at(j));
				temp1 = split.at(i);
				hconcat(temp1, temp, temp1);
				split.at(i) = temp1;
			}


		}
	}
	return split;
}


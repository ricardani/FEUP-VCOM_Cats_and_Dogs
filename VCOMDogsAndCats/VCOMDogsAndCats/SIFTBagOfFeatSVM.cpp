// VCOMDogsAndCats.cpp : Defines the entry point for the console application.
// This file test machine learning using SIFT for local features detection and extraction,bag of features for vocabulary and SVM for machine learning
//1st try (SIFT+100+10 clusters+SVM.kernel linear)  0.43851
//2nd try (SIFT+300+10 clusters+SVM.kernel linear) 0.55
//3rd try (SURF+300+10 clusters+SVM.kernel linear) 0.49634
//4th try (SIFT+300+20 clusters+SVM.kernel linear) 0.44320
//5th try (SIFT+300+20 clusters+SVM.kernel RBF) 0.45589
//6th try (SIFT+300+10 clusters+SVM.kernel RBF) 0.50400
//7th try (SIFT+1000+10 clusters+SVM.kernel linear) 0.52114
//8th try (SIFT+300+5 clusters+SVM.kernel linear) 	0.53749
//9th try (SIFT+300+100 clusters+SVM.kernel linear+trainAuto prof recomendation) 0.51691
//10th try (SIFT+1000+100 clusters+SVM.kernel linear+trainAuto prof recomendation) 0.52183
#include "stdafx.h"
/*
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/core/core.hpp>
#include <fstream>
#include <iostream>
#include <omp.h>

using namespace cv;
using namespace std;
using namespace ml;

Ptr<FeatureDetector> detector = xfeatures2d::SIFT::create();
Ptr<DescriptorExtractor> extractor = xfeatures2d::SIFT::create();
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
int numberClusters = 10;
int numberOfImages = 300;//300 cats 300 dogs
BOWKMeansTrainer bowTrainer(numberClusters);
BOWImgDescriptorExtractor bowDE(extractor, matcher);


bool openImage(const std::string &filename, Mat &image)
{
	image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	if (!image.data) {
		cout << " --(!) Error reading image " << filename << std::endl;
		return false;
	}
	return true;
}

//gets the name of the images
vector<String> getImagesNames()
{
	vector<String> listOfImages = vector<String>();
	for (int i = 0;i < numberOfImages;i++)
	{
		listOfImages.push_back(("train/cat." + to_string(i) + ".jpg"));
	}
	for (int i = 0;i < numberOfImages;i++)
	{
		listOfImages.push_back(("train/dog." + to_string(i) + ".jpg"));
	}
	return listOfImages;
}

void trainBagOfFeatures(vector<String> listOfImages)
{
	Mat image;
	vector<KeyPoint> keypoints;
	Mat descriptors, allDescriptors;

	for (int i = 0; i < listOfImages.size(); i++)
	{
		if (!openImage(listOfImages[i], image))
			continue;

		cout << i << endl;
		detector->detect(image, keypoints);
		extractor->compute(image, keypoints, descriptors);
		allDescriptors.push_back(descriptors);
	}

	//num clusters
	bowTrainer.add(allDescriptors);
	Mat vocabulary = bowTrainer.cluster();

	FileStorage fs1("voc.yml", FileStorage::WRITE);
	fs1 << "vocabulary" << vocabulary;
	fs1.release();

	Mat dictionary = bowTrainer.cluster();
	bowDE.setVocabulary(dictionary);
}

void trainSVM(vector<String> listOfImages)
{
	Mat image;

	// Set up SVM's parameters
	Ptr< SVM > svm = SVM::create();

	// Train the SVM
	Mat labels(0, 1, CV_32FC1);
	Mat trainingData(0, numberClusters, CV_32FC1);
	vector<KeyPoint> keypoint;
	Mat bowDescriptor;

	//cats
	for (int i = 0;i < numberOfImages;i++) {


		if (!openImage(listOfImages[i], image))
			continue;

		cout << i << endl;

		detector->detect(image, keypoint);
		bowDE.compute(image, keypoint, bowDescriptor);
		trainingData.push_back(bowDescriptor);

		labels.push_back(0);
	}

	//dogs
	for (int i = numberOfImages;i < numberOfImages * 2;i++) {


		if (!openImage(listOfImages[i], image))
			continue;

		cout << i << endl;

		detector->detect(image, keypoint);
		bowDE.compute(image, keypoint, bowDescriptor);
		trainingData.push_back(bowDescriptor);//add histogram

		labels.push_back(1);
	}

	printf("%s\n", "Training SVM classifier");

	Ptr<TrainData> td = TrainData::create(trainingData, ROW_SAMPLE, labels);
	//bool res = svm->train(trainingData, ROW_SAMPLE, labels);
	//cout << res;
	svm->trainAuto(td, 100);
	svm->save("test.xml");
}

void train(bool trainVoc)
{
	vector<String> listOfImages = vector<String>();
	listOfImages = getImagesNames();
	if (trainVoc)
	{
	trainBagOfFeatures(listOfImages);
	}
	else
	{
	FileStorage fs("voc.yml", FileStorage::READ);
	if (fs.isOpened())
	{
	fs["vocabulary"] >> dictionary;
	}
	bowDE.setVocabulary(dictionary);
	}

	trainSVM(listOfImages);

}

void testImages()
{
	Mat image, descriptors, dictionary, allDescriptors = Mat();
	vector<KeyPoint> keypoints;

	//load trained machine
	Ptr< SVM > svm = SVM::load<SVM>("test.xml");

	FileStorage fs("voc.yml", FileStorage::READ);
	if (fs.isOpened())
	{
		fs["vocabulary"] >> dictionary;
	}
	bowDE.setVocabulary(dictionary);
	//end of load

	ofstream rspCSV("rsp.csv");
	// this does the open for you, appending data to an existing file
	rspCSV << "id,label" << endl;

	for (int i = 1; i <= 12500; i++)
	{
		if (!openImage("test1/" + to_string(i) + ".jpg", image))
			continue;

		cout << i << endl;
		detector->detect(image, keypoints);
		bowDE.compute(image, keypoints, descriptors);
		rspCSV << i << "," << svm->predict(descriptors) << std::endl;

	}
	rspCSV.close();
	waitKey(0);

}


int main()
{
//0->complete training
//1->train only machine learning
//2->test
int mode = 0;
if (mode==0)
{
train(true);
}
else if (mode == 1)
{
train(false);
}
else
{
testImages();
}
return 0;
}

*/
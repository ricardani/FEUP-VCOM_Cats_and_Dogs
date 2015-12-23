// VCOMDogsAndCats.cpp : Defines the entry point for the console application.
// This file test machine learning using SIFT for local features detection and extraction,bag of features for vocabulary and SVM, Neural Networks and Bayes for machine learning
#include "stdafx.h"

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

Ptr<FeatureDetector> detector = xfeatures2d::SIFT::create();//creates a SIFT detector
Ptr<DescriptorExtractor> extractor = xfeatures2d::SIFT::create();//creates a SIFT extractor
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
int numberClusters = 400;
int numberOfImages = 1500;//1500 cats 1500 dogs
BOWKMeansTrainer bowTrainer(numberClusters);//trains the bag of features with k-means algorithm
BOWImgDescriptorExtractor bowDE(extractor, matcher);//bag of features extractor
Ptr< SVM > svm = SVM::create();
Ptr< Boost> boost = Boost::create();
Ptr<KNearest> kn = KNearest::create();



bool openImage(const std::string &filename, Mat &image)
{
	image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	if (!image.data) {
		cout << " --(!) Error reading image " << filename << std::endl;
		return false;
	}
	return true;
}

//returns a vector with the name of the train images
vector<String> getImagesNames()
{
	vector<String> listOfImages = vector<String>();
	for (int i = 0; i < numberOfImages; i++)
	{
		listOfImages.push_back(("train/cat." + to_string(i) + ".jpg"));
	}
	for (int i = 0; i < numberOfImages; i++)
	{
		listOfImages.push_back(("train/dog." + to_string(i) + ".jpg"));
	}
	return listOfImages;
}

//trains bag of features with the images given to it
void trainBagOfFeatures(vector<String> listOfImages)
{
	Mat image;
	vector<KeyPoint> keypoints;
	Mat descriptors, allDescriptors;

	//Extract descriptors from all images
	for (int i = 0; i < listOfImages.size(); i++)
	{
		if (!openImage(listOfImages[i], image))
			continue;

		cout << i << endl;
		detector->detect(image, keypoints);
		extractor->compute(image, keypoints, descriptors);
		allDescriptors.push_back(descriptors);
	}

	//Create vocabulary
	bowTrainer.add(allDescriptors);
	Mat vocabulary = bowTrainer.cluster();

	FileStorage fs1("voc.yml", FileStorage::WRITE);
	fs1 << "vocabulary" << vocabulary;
	fs1.release();

	//Define Dictionary
	bowDE.setVocabulary(vocabulary);
}

void trainML(vector<String> listOfImages)
{
	Mat image;

	Mat labels(0, 1, CV_32FC1);
	Mat trainingData(0, numberClusters, CV_32FC1);
	vector<KeyPoint> keypoint;
	Mat bowDescriptor;

	//Extract descriptors of cats
	for (int i = 0; i < numberOfImages; i++) {


		if (!openImage(listOfImages[i], image))
			continue;

		cout << i << endl;

		detector->detect(image, keypoint);
		bowDE.compute(image, keypoint, bowDescriptor);
		trainingData.push_back(bowDescriptor);

		labels.push_back(0);
	}

	//Extract descriptors of dogs
	for (int i = numberOfImages; i < numberOfImages * 2; i++) {


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

	//Train SVM, K-NN and Boosting
	svm->trainAuto(td, 100);
	boost->train(td);
	kn->train(td);

	//saves the trained machines
	svm->save("svm.xml");
	boost->save("boost.xml");
	kn->save("kn.xml");
}

//train 
void train(bool trainVoc)
{
	vector<String> listOfImages = vector<String>();
	listOfImages = getImagesNames();
	Mat dictionary;
	//Create vocabulary if necessary
	if (trainVoc)
	{
		trainBagOfFeatures(listOfImages);
	}
	else
	{
		try
		{
			FileStorage fs("voc.yml", FileStorage::READ);
			if (fs.isOpened())
			{
				fs["vocabulary"] >> dictionary;
			}
			bowDE.setVocabulary(dictionary);
		}
		catch(exception& e)
		{
			cerr << e.what() << '\n';
		}
	}


	trainML(listOfImages);

}

//tests all the images in the folder test1 and writes the results of which machine and the result of the 3 combined.
void predictTestSet()
{
	Mat image, descriptors, dictionary, allDescriptors = Mat();
	vector<KeyPoint> keypoints;

	try
	{
		//load trained machine
		svm = SVM::load<SVM>("svm.xml");
		boost = SVM::load<Boost>("boost.xml");
		kn = SVM::load<KNearest>("kn.xml");

		//load vocabulary for bag of features
		FileStorage fs("voc.yml", FileStorage::READ);
		if (fs.isOpened())
		{
			fs["vocabulary"] >> dictionary;
		}
		bowDE.setVocabulary(dictionary);
		//end of load

		//open csv files
		ofstream rspCSV("rsp.csv");
		ofstream svmCSV("SVMrsp.csv");
		ofstream boostCSV("Boostrsp.csv");
		ofstream knnCSV("KNNrsp.csv");

		//first line of which file
		rspCSV << "id,label" << endl;
		svmCSV << "id,label" << endl;
		boostCSV << "id,label" << endl;
		knnCSV << "id,label" << endl;

		for (int i = 1; i <= 12500; i++)
		{
			if (!openImage("test1/" + to_string(i) + ".jpg", image))
				continue;

			cout << i << endl;
			detector->detect(image, keypoints);
			bowDE.compute(image, keypoints, descriptors);

			//Predict of which machine
			float svmP = svm->predict(descriptors);
			float boostP = boost->predict(descriptors);
			float knP = kn->predict(descriptors);
			int r = round((svmP + boostP + knP) / 3.0);

			rspCSV << i << "," << r << endl;
			svmCSV << i << "," << svmP << endl;
			boostCSV << i << "," << boostP << endl;
			knnCSV << i << "," << knP << endl;
		}
		rspCSV.close();
		svmCSV.close();
		boostCSV.close();
		knnCSV.close();
	}
	catch (exception& e)
	{
		cerr << e.what() << '\n';
	}
}

//tests one image to see if it is a dog or a cat
void predictImage()
{
	Mat image, descriptors, dictionary, allDescriptors = Mat();
	vector<KeyPoint> keypoints;

	string image_name;
	cout << "Image: " << endl;
	cin >> image_name;


	//load trained machine
	svm = SVM::load<SVM>("svm.xml");
	boost = SVM::load<Boost>("boost.xml");
	kn = SVM::load<KNearest>("kn.xml");

	FileStorage fs("voc.yml", FileStorage::READ);
	if (fs.isOpened())
	{
		fs["vocabulary"] >> dictionary;
	}
	else 
	{
		cout << "Failed to open voc.yml!" << endl;
		return;
	}

	bowDE.setVocabulary(dictionary);
	//end of load

	if (!openImage(image_name, image)) {
		cout << "Failed to open image: " << image_name << endl;
		return;
	}

	detector->detect(image, keypoints);
	bowDE.compute(image, keypoints, descriptors);

	float svmP = svm->predict(descriptors);
	float boostP = boost->predict(descriptors);
	float knP = kn->predict(descriptors);
	int r = round((svmP + boostP + knP) / 3.0);

	if (r == 0)
		cout << "This is a Cat!" << endl;
	else 
		cout << "This is a Dog!" << endl;

}

void menu() {
	string option;
	int MENU_OPTION;

	cout << "  ____   ___   ____ ____                 ____    _  _____ ____  " << endl;
	cout << " |  _ \\ / _ \\ / ___/ ___|  __   _____   / ___|  / \\|_   _/ ___| " << endl;
	cout << " | | | | | | | |  _\\___ \\  \\ \\ / / __| | |     / _ \\ | | \\___ \\ " << endl;
	cout << " | |_| | |_| | |_| |___) |  \\ V /\\__ \\ | |___ / ___ \\| |  ___) |" << endl;
	cout << " |____/ \\___/ \\____|____/    \\_/ |___/  \\____/_/   \\_\\_| |____/ " << endl;
	cout << endl << endl;

	cout << "Choose an option:" << endl;
	cout << "1 - Complete Training" << endl;
	cout << "2 - Train With voc.yml" << endl;
	cout << "3 - Predict Test Set" << endl;
	cout << "4 - Predict Image" << endl;
	cout << "0 - Exit" << endl;
	cin >> option;

	MENU_OPTION = atoi(option.c_str());

	switch (MENU_OPTION)
	{
	case 1:
		train(true);
		break;
	case 2:
		train(false);
		break;
	case 3:
		predictTestSet();
		break;
	case 4:
		predictImage();
		break;
	default:
		return;
		break;
	}

}


int main()
{
	menu();
	return 0;
}


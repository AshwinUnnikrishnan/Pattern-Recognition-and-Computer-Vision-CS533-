#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int readImagesExtractTaskOne(string path, string csvName); //Reads all the images from the path and extracts the feature to the csvName provided
//int can be used to return the total number of images read in future
vector<string>  readBackImageTaskOne(string imagePath, string path, int topMatchCount);
//Takes imagePath where the image to compare is, path of all the other images and then number of results to show as input and returns the topn N similar image paths
int readImagesExtractTaskTwo(string path, string csvName); //Reads all the images from the path and extracts the feature to the csvName provided

vector<string>  readBackImageTaskTwo(string imagePath, string path, int topMatchCount);
//Takes imagePath where the image to compare is, path of all the other images and then number of results to show as input and returns the topn N similar image paths

int readImagesExtractTaskTwoExtension(string path, string csvName); //Reads all the images from the path and extracts the feature to the csvName provided

vector<string>  readBackImageTaskTwoExtension(string imagePath, string path, int topMatchCount);
//Takes imagePath where the image to compare is, path of all the other images and then number of results to show as input and returns the topn N similar image paths

int readImagesExtractTaskBlueBin(string path, string csvName); //Reads all the images from the path and extracts the feature to the csvName provided

vector<string>  readBackImageTaskBlueBin(string imagePath, string path, int topMatchCount);
//Takes imagePath where the image to compare is, path of all the other images and then number of results to show as input and returns the topn N similar image paths

int readImagesExtractTaskThree(string path, string csvName); //Reads all the images from the path and extracts the feature to the csvName provided

vector<string>  readBackImageTaskThree(string imagePath, string path, int topMatchCount);
//Takes imagePath where the image to compare is, path of all the other images and then number of results to show as input and returns the topn N similar image paths

int readImagesExtractTaskFour(string path, string csvName); //Reads all the images from the path and extracts the feature to the csvName provided

vector<string>  readBackImageTaskFour(string imagePath, string path, int topMatchCount);
//Takes imagePath where the image to compare is, path of all the other images and then number of results to show as input and returns the topn N similar image paths

int readImagesExtractTaskFive(string path, string csvName); //Reads all the images from the path and extracts the feature to the csvName provided

vector<string>  readBackImageTaskFive(string imagePath, string path, int topMatchCount);
//Takes imagePath where the image to compare is, path of all the other images and then number of results to show as input and returns the topn N similar image paths

int readImagesExtractTaskFourExtension(string path, string csvName); //Reads all the images from the path and extracts the feature to the csvName provided

vector<string>  readBackImageTaskFourExtension(string imagePath, string path, int topMatchCount);
//Takes imagePath where the image to compare is, path of all the other images and then number of results to show as input and returns the topn N similar image paths

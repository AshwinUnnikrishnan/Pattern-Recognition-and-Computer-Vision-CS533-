#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <wx/wx.h>
using namespace cv;

using namespace std;

int bubbleSortIndex(vector<float> &distanceMetric, vector<string> &imageIndex);

float sumOfSquares(vector<float> src, vector<float> dest);

float chiSquare(vector<float> src, vector<float> dest);

float manHattanDist(vector<float> src, vector<float> dest);

string pathToCompatibleString(string pathV);

int greyScale(Mat &src, Mat &dst);

int blur5x5(Mat &src, Mat &dst);

int sobelX3x3( Mat &src, Mat &dst );

int sobelY3x3( Mat &src, Mat &dst );

int magnitudeSobel( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst );

int sobelY3x3nonSeparable( Mat &src, Mat &dst );

int sobelX3x3NonSeparable( Mat &src, Mat &dst );

int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels );

int brightnessIncrease(Mat &src, Mat &dst,  int brightness);

int laplaceFilter(Mat &src, Mat &dst);

int cartoon( cv::Mat &src, cv::Mat&dst, int levels, int magThreshold );

//int putLegendText( cv::Mat &src, cv::Mat &dst, std::vector<String> stats);
int putLegendText( cv::Mat &src, cv::Mat &dst, std::vector<String> stats);


int sharpeness( cv::Mat &src, cv::Mat &dst);

int reSize( cv::Mat &src, cv::Mat &dst, float dx = 0.7, float dy = 0.7);

int colorPalleteChange(cv::Mat &src, cv::Mat &dst);

int medianFilter(cv::Mat &src, cv::Mat &dst);

int lawsFilterE5L5( Mat &src, Mat &dst );

Mat readData(String filenameI);

int thresholdBinary( Mat &src, Mat &dst, int flag);

int binaryCleanUp( Mat &src, Mat &dst, int size_ker, int flag=0);

int grow_shrink(Mat &src, Mat &dst, int iteration, int size_ker, int flag=0);

int conectedCompo(Mat &src, Mat &dst, int regionFlag, vector<vector<double>> &huMomentsMain);

string findLabelMost(vector<string>);
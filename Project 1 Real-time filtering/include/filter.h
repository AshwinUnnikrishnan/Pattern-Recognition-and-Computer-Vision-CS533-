#include<includeFile.h>
using namespace cv;

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

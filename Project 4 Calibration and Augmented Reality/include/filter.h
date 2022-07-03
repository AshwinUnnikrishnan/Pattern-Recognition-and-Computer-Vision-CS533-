#include<includeFile.h>
using namespace cv;
using namespace std;


char * strToChar(String st);

int saveImageAndFeature(Mat &src,  Mat &updImg,vector<Point2f> corner_set, vector<Vec3f> point_set,std::string fileName, int count);


int buildCameraMatrix(float cols, float rows, Mat &camM, double focalLength=1);

String printMat(Mat src, String str);

int writeParam(String cameraName, Mat cameraMat, Mat distCoeff);

int drawAxes(Mat &src, Mat rvec1, Mat tvec1, Mat cameraMat, Mat distCoeffs, Point2f corner);

void cornerHarris_d( Mat &src, Mat &dst);

void myShiTomasi_function( Mat &src, Mat &dst);

void drawRectangle(Mat &colorFrame, std::vector<cv::Point2f> corners, std::vector<cv::Point2f> corners_2);

int checkandLoadCalib(String csvFileName, String cameraName, Mat &cameraMat, Mat &distCoeffs);

void hideBackgroundFunc(Mat &colorFrame);
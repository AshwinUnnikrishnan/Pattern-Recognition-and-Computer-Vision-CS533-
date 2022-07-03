#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;


const char* source_window = "Source image";
const char* corners_window = "Corners detected";

void cornerHarris_testing(Mat &src, Mat &dst);

int main( int, char** argv )
{
string image_path = samples::findFile("chess.jpg",false);
Mat src,dst;

  src = cv::imread(image_path,IMREAD_COLOR);

  namedWindow( source_window, WINDOW_AUTOSIZE );
  //createTrackbar( "Threshold: ", source_window, &thresh,200,0 );

  imshow( source_window, src );

  cornerHarris_testing(src, dst);
  namedWindow( corners_window, WINDOW_AUTOSIZE );
  imshow( corners_window, dst );
  waitKey(0);
  return(0);
}

int cornerHarris_testing(Mat &src, Mat &dst)
{

  Mat dst_norm, src_gray;
  dst = Mat::zeros( src.size(), CV_32FC1 );
  int thresh = 150;
  int max_thresh = 255;

  int blockSize = 2;
  int apertureSize = 3;
  double k = 0.04;
  cvtColor( src, src_gray, COLOR_BGR2GRAY );

  cornerHarris( src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT );

  normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
  convertScaleAbs( dst_norm, dst);

  for( int j = 0; j < dst_norm.rows ; j++ )
     { for( int i = 0; i < dst_norm.cols; i++ )
          {
            if( (int) dst_norm.at<float>(j,i) > thresh )
              {
               circle( dst, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
              }
          }
     }
  
}


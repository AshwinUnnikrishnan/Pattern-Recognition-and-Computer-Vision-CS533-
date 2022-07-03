#include<includeFile.h>
#include<filter.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <filesystem>
#include "csv_util.cpp"
#include <dirent.h>
#include <iostream>

using namespace std;
using namespace cv;
char * strToChar(String st){
    char *cam = new char[st.length() + 1];
    strcpy(cam, st.c_str());
    return cam;
}


int saveImageAndFeature(Mat &src, Mat &updImg, std::vector<Point2f> corner_set, vector<Vec3f> point_set, std::string fileName, int count){
    /*
        Gets the Mat image and then saves the image and chessboardDetected Image in the folder
        src : Captured image
        updImg : chessBoard detected Image
        filename : string chess or something else need to convert to image and csv names
        count : count of the number of images stored in current run
    */
    if (mkdir(strToChar(fileName), 0777) == -1)
        cout<<"chessBoard directory Already Exists"<<endl;
    else
        cout << "chessBoard Directory created";
    imwrite(fileName + "/"+fileName+to_string(count)+".jpg", src); // A JPG FILE IS BEING SAVED
    imwrite(fileName + "/Update"+fileName+to_string(count)+".jpg", updImg); // A JPG FILE IS BEING SAVED
    return 1;
}

int buildCameraMatrix(float width, float height, Mat &camM, double focalLength){
    /*
        Calculates the cameramatrix for the given rows and columns and stores in the camM
    */
    Mat camMatrix(cv::Size(3, 3), CV_64FC1);
    camMatrix = 0;
    camMatrix.at<double>(0,0) = camMatrix.at<double>(1,1) = focalLength;
    camMatrix.at<double>(2,2) = 1;
    camMatrix.at<double>(0,2) = width;
    camMatrix.at<double>(1,2) = height;
    camM = camMatrix.clone();
    return 1;
}

String printMat(Mat src, String str){
    /*
        Given a  Mat prints the string and then contents of the Mat
    */
    String res("Mat " + str);
    cout<<res<<endl;
    for(int x = 0; x < src.rows; x++){
        for(int y = 0; y < src.cols; y++){
            cout<<src.at<double>(x,y)<<"          ";
            res = res + to_string(src.at<double>(x,y)) + "          ";
        }
        cout<<endl;
    }
    return res;
}

int writeParam(String cameraName, Mat cameraMat, Mat distCoeff){
    /*
        Writes the intrinsic parameter to file
    */
    // Adding focal lengths, u0,v0, distcoefficients to a float vector then storing it into the csv file
    vector<float> temp;
    temp.push_back(cameraMat.at<double>(0,0));   //fx
    temp.push_back(cameraMat.at<double>(1,1));   //fy
    temp.push_back(cameraMat.at<double>(0,2));   //cx
    temp.push_back(cameraMat.at<double>(1,2));   //cy
    for(int x = 0; x < distCoeff.rows; x++){
        for(int y = 0; y < distCoeff.cols; y++){
            temp.push_back(distCoeff.at<double>(x,y));   //k1 k2 p1 p2 k3
        }
    }
    append_image_data_csv(strToChar(cameraName+".csv"), strToChar(cameraName), temp, 1);
    return 1;
}


int drawAxes(Mat &src, Mat rvec1, Mat tvec1, Mat cameraMat, Mat distCoeffs, Point2f originC){
    /*
        Calculates 3D points with respect to origin and then draws the 3D axes of 2 Unit length
    */
    std::vector<cv::Vec3f> axesPoints;
    axesPoints.push_back(Point3f(2,0,0));
    axesPoints.push_back(Point3f(0,2,0));
    axesPoints.push_back(Point3f(0,0,2));
    std::vector<cv::Point2f> axes;

    projectPoints(axesPoints, rvec1, tvec1, cameraMat, distCoeffs, axes); //gets all the corners in 3D
    line(src, originC, axes[0], Scalar(0, 255, 0),6, LINE_4);                        
    line(src, originC, axes[1], Scalar(255, 255, 0),6, LINE_4);                        
    line(src, originC, axes[2], Scalar(0, 255, 255),6, LINE_4);                        
    return 1;
}


void cornerHarris_d( Mat &src, Mat &dst)
{
    /*
        Harris Corner Detection Algorithm 
    */
    Mat dst_norm, src_gray;
    dst = Mat::zeros( src.size(), CV_32FC1 );
    int thresh = 150;
    int max_thresh = 255;

    int blockSize = 5;
    int apertureSize = 10;
    double k = 0.04;
    cvtColor( src, src_gray, COLOR_BGR2GRAY );

    cornerHarris( src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT );

    normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs( dst_norm, dst);

    for( int j = 0; j < dst_norm.rows ; j++ ){
        for( int i = 0; i < dst_norm.cols; i++ ){
            if( (int) dst_norm.at<float>(j,i) > thresh ){
                circle( dst, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
                }
            }
        }
}


void myShiTomasi_function( Mat &src, Mat &dst)
{
    /*
        ShiTomasi Corner Detection Algorithm
    */
    RNG rng(12345);
    Mat src_gray;
    int blockSize = 2;
    int apertureSize = 3;
    int max_qualityLevel = 100;
    int myShiTomasi_qualityLevel = 50;

    double myShiTomasi_minVal, myShiTomasi_maxVal;

    cvtColor( src, src_gray, COLOR_BGR2GRAY );

    Mat myShiTomasi_copy, myShiTomasi_dst;

    cornerMinEigenVal( src_gray, myShiTomasi_dst, blockSize, apertureSize );
    minMaxLoc( myShiTomasi_dst, &myShiTomasi_minVal, &myShiTomasi_maxVal );


    myShiTomasi_copy = src.clone();
    myShiTomasi_qualityLevel = 50;
    for( int i = 0; i < src_gray.rows; i++ )
    {
        for( int j = 0; j < src_gray.cols; j++ )
        {
            if( myShiTomasi_dst.at<float>(i,j) > myShiTomasi_minVal + ( myShiTomasi_maxVal - myShiTomasi_minVal )*myShiTomasi_qualityLevel/max_qualityLevel )
            {
                circle( myShiTomasi_copy, Point(j,i), 4, Scalar( rng.uniform(0,256), rng.uniform(0,256), rng.uniform(0,256) ), FILLED );
            }
        }
    }
    dst = myShiTomasi_copy.clone();
}

void hideBackgroundFunc(Mat &colorFrame){
    /*
        Making all the elements of the screen black and then drawing only the virtual object
    */
    for(int x = 0; x < colorFrame.rows; x++){
        for(int y = 0; y < colorFrame.cols; y++){
            Vec3b destColor = colorFrame.at<Vec3b>(x,y);
            destColor.val[0] = destColor.val[1] = destColor.val[2] = 0;
            colorFrame.at<Vec3b>(x,y) = destColor;
        }
    }
}

void colorSides(Mat &src, Point tl, Point tr, Point br, Point bl, int color){
    /*
        Given four sides of a rectangle colors the sieds
    */
    std::vector<cv::Point> fillContSingle;
    fillContSingle.push_back(tl);
    fillContSingle.push_back(tr);
    fillContSingle.push_back(br);
    fillContSingle.push_back(bl);
    std::vector<std::vector<cv::Point> > fillContAll;
    fillContAll.push_back(fillContSingle);
    cv::fillPoly( src, fillContAll, cv::Scalar(color));
}

void drawRectangle(Mat &colorFrame, std::vector<cv::Point2f> corners, std::vector<cv::Point2f> corners_2){
    /*
        Draws a cuboid with the corner points given
    */
    colorSides(colorFrame, corners[0], corners[8], corners_2[8], corners_2[0], 50);
    colorSides(colorFrame, corners[0], corners[45], corners_2[45], corners_2[0], 100);
    colorSides(colorFrame, corners[45], corners[53], corners_2[53], corners_2[45], 140);
    colorSides(colorFrame, corners[53], corners[8], corners_2[8], corners_2[53], 200);

    
    line(colorFrame, corners[0], corners_2[0], Scalar(255, 255, 0),6, LINE_4);      
    line(colorFrame, corners_2[0], corners_2[8], Scalar(255, 255, 0),6, LINE_4);
    line(colorFrame, corners[0], corners[8], Scalar(255, 255, 0),6, LINE_4);
    line(colorFrame, corners[0], corners[45], Scalar(255, 255, 0),6, LINE_4);                        
    line(colorFrame, corners_2[0], corners_2[45], Scalar(255, 255, 0),6, LINE_4);                        

    line(colorFrame, corners[8], corners_2[8], Scalar(255, 255, 0),6, LINE_4);  
    line(colorFrame, corners[8], corners[53], Scalar(255, 255, 0),6, LINE_4);
    line(colorFrame, corners_2[8], corners_2[53], Scalar(255, 255, 0),6, LINE_4);


    line(colorFrame, corners[45], corners_2[45], Scalar(255, 255, 0),6, LINE_4);                        
    line(colorFrame, corners[53], corners_2[53], Scalar(255, 255, 0),6, LINE_4);      

    line(colorFrame, corners[45], corners[53], Scalar(255, 255, 0),6, LINE_4);
    line(colorFrame, corners_2[45], corners_2[53], Scalar(255, 255, 0),6, LINE_4);
    imwrite("Fillind3D.jpg", colorFrame);

}

int checkandLoadCalib(String csvFileName, String cameraName, Mat &cameraMat, Mat &distCoeffs){
    /*
        Checks and loads calibration data into the cameraMat and distCoeffs
    */
    char csvName[csvFileName.length()];
    strcpy(csvName, csvFileName.c_str());
    vector<char *> cameraNames;
    char camName[cameraName.length()];
    strcpy(camName, cameraName.c_str());

    std::vector<std::vector<float> > cameraProperties;
    read_image_data_csv(csvName, cameraNames, cameraProperties, 0);
    int i;
    for(i = 0;i < cameraNames.size(); i++){
        if( strcmp(cameraNames[i], camName) == 0){
            break;
        }
    }
    //first two would be focal length followed by width and height in the cameraproperties vector followed by the distcoeff
    //buildCameraMatrix(cameraProperties[2], cameraProperties[3], cameraMat, cameraProperties[0]);
    buildCameraMatrix(cameraProperties[i][2], cameraProperties[i][3], cameraMat, cameraProperties[i][0]);
    printMat(cameraMat, "Loaded perfectly");
    for(int j = 4; j<=8 ; j++){
        distCoeffs.at<double>(j-4,0)= cameraProperties[i][j];
    }

    printMat(distCoeffs, "Distortion Coefficient After calibration"); 
    return 1;
}

#include <wx/filepicker.h>
#include <wx/sysopt.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <string>
#include <filter.h>

#include <wx/wxprec.h>
#ifndef WX_PRECOMP
#include <wx/wx.h>
#endif

using namespace cv;
using namespace std;
#include <wx/wx.h>
#include <cmath>


string findLabelMost(vector<string> words){
    /*
        So from the top 10 closest neighbours this function finds the label most repeated and then returns the label
        Can improve on this by finding the sum distance again and then finding the least
    */
    int num[10] = { 0, 0, 0, 0, 0,0,0,0,0,0};
    for(int j = 0; j < 10; j++)
    {
        string t1 = words[j];
        for(int i = 0; i < 10; i++)
                if(t1 == words[i])
                        num[j]++;
    }
    int maxIndex = 0;
    for(int i=1 ; i<10; i++){
        if(num[i] > num[maxIndex]){
            maxIndex = i;
        }
    }
    return words[maxIndex];
}

int bubbleSortIndex(vector<float> &distanceMetric, vector<string> &imageName){
    /*
    Modified bubble sort, where the sorting happens for 10 elements and the bubble sort gets called for N images
    So the complexity is good O(N*10) i.e. O(N), storing the value and the image index
    */
    for(int k = 0; k < distanceMetric.size()-1; k++){
        for(int z = 0 ; z < distanceMetric.size()-k-1; z++){
            if(distanceMetric[z] > distanceMetric[z+1]){
                double temp = distanceMetric[z];
                string tempIndex = imageName[z];
                distanceMetric[z] = distanceMetric[z+1];
                imageName[z] = imageName[z+1];
                distanceMetric[z+1] = temp;
                imageName[z+1] = tempIndex;
            }
        }
    }
    return 0;
}

string pathToCompatibleString(string pathV){
    /*
        To convert string to path string format / -> //
    */
    std::replace(pathV.begin(), pathV.end(), '/', '|');
    size_t pos;
    while ((pos = pathV.find("|")) != std::string::npos) {
        pathV.replace(pos, 1, "//");
    }
    return pathV;
}

float manHattanDist(vector<float> src, vector<float> dest){
    /*
        manHattanDistance metric calculation
    */
    float sum = 0.0;
    for(int i = 0; i < src.size() ; i++){
        sum += abs(src[i] - dest[i]);
    }
    return sum;
}

float sumOfSquares(vector<float> src, vector<float> dest){
    /*
    Sum of Squre Distance Metric Calculation
    */
    float sum = 0.0;
    for(int i = 0; i < src.size() ; i++){
        sum += (src[i] - dest[i]) * (src[i] - dest[i]);
    }
    //cout<<sum;
    return sqrt(sum);
}

float chiSquare(vector<float> src, vector<float> dest){
    /*
    Chi Square Distance
    */
    float sum = 0.0;
    for(int i = 0; i < src.size() ; i++){
        sum += ((src[i] - dest[i]) * (src[i] - dest[i]))/(src[i] + dest[i]);
    }
    //cout<<sum;
    return sum/2;
}

int greyScale(Mat &src, Mat &dst){
    /*
    This is inverted greyscale implementation where any value in range 200-250 in RGB channel then the pixel is made black that is colors close to white is made black
    and other colors are made white
    */
    dst = src.clone();
    for(int x = 0; x < src.rows; x++){
        for(int y = 0; y < src.cols; y++){
            Vec3b color = src.at<Vec3b>(x,y);
            //uchar avg = (uchar) (0.0722 * color.val[0] + 0.7152 * color.val[1] + 0.2126 * color.val[2]);
            Vec3b destColor = dst.at<Vec3b>(x,y);

            destColor.val[0] = destColor.val[1] = destColor.val[2] = 0;

            if(color.val[0] > 200 || color.val[2] > 200 || color.val[1] > 200){
                destColor.val[0] = destColor.val[1] = destColor.val[2] = 0;
            }
            else{
                destColor.val[0] = destColor.val[1] = destColor.val[2] = 255;
            }
            dst.at<Vec3b>(x,y) = destColor;
        }
    }
    return 0;
}

int thresholdBinary( Mat &src, Mat &dst, int flag){
    /*
        Binary Thresholding
        Converting it to greyScale and making pixels black below thresh and white above thresh
        flag : 1) Binary Inverse Threshold 2)HSV thresholding 3) ownImplementation 4) OTSU thresholding
    */
    Mat temp, hsv, out;
    switch(flag){
        case 1:
            cvtColor(src, temp, COLOR_BGR2GRAY);
            threshold(temp, dst, 150, 255, THRESH_BINARY_INV);//converting grayscale image stored in converted matrix into binary image//
            break;
        case 2:
            cvtColor(src, temp, COLOR_BGR2HSV);
        // Detect the object based on HSV Range Values
            inRange(temp, Scalar(0, 0, 0), Scalar(255, 255, 210), dst);
            break;
        case 3:
            greyScale(src, dst);
            break;
        case 4:
            cvtColor(src, temp, COLOR_BGR2GRAY);
            threshold(temp, dst, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
            break;
    }
    return 1;
}

int binaryCleanUp( Mat &src, Mat &dst, int size_ker, int flag){
    /*
        Performs shrink and grow based on the size_ker
    */

    Mat element = getStructuringElement( MORPH_RECT, Size(size_ker, size_ker) );
    if(flag == 0){
        dilate(src, dst, element);
    }
    else{
        erode(src, dst, element);
    }
    return 1;
}

int grow_shrink(Mat &src, Mat &dst, int iteration, int size_ker, int flag){
    /*
        Performs the grow/shrink iteration number of times
    */
    Mat* L = new Mat[iteration+1]; 
    L[0] = src.clone();
    for(int i=0;i<iteration;i++){
        binaryCleanUp(L[i], L[i+1], size_ker, flag);
    }
    dst = L[iteration].clone();
    delete[] L;
    return 1;
}

int conectedCompo(Mat &src, Mat &dst, int regionFlag, vector<vector<double>> &huMomentsMain){
    /*
        If region flag is set then it will return the region colored image, 3 colors are used red, green, blue to color the different regions alternatively
        If region flag is unset then for each region identified it identifies all the points in that region and then draws a oriented bounding box and the centroid around the regions and then returns the respective humoments
    */
    Mat labelImage(src.size(), CV_32S);
    Mat stats;
    Mat centroids;
    dst = src.clone();

    int nLabels = connectedComponentsWithStats(src, labelImage, stats, centroids);
    dst = src.clone();
    std::vector<cv::Moments> mu(stats.rows - 1);
    vector<vector<double>> huMoments(stats.rows - 1);
    if(regionFlag == 0){
        for(int i=1; i<stats.rows; i++)//skipping the background
        {
            int x = stats.at<int>(Point(0, i));
            int y = stats.at<int>(Point(1, i));
            int w = stats.at<int>(Point(2, i));
            int h = stats.at<int>(Point(3, i));
            if((w*h) < 1000){
                continue;
            }
            std::vector<cv::Point> points;
            for(int co = x; co < x+w ; co++){
                for(int ro = y; ro < y+h ; ro++ ){
                    if((int)(src.at<uchar>(cv::Point2i(co,ro))) == 255){
                        points.push_back(Point(co,ro));
                    }
                }
            }
            if(points.size() != 0){
                RotatedRect box = minAreaRect(cv::Mat(points));
                cv::Point2f vertices[4];
                box.points(vertices);

                for(int j = 0; j < 4; ++j)
                    cv::line(dst, vertices[j], vertices[(j + 1) % 4], cv::Scalar(255, 0, 0), 1, 8);

                double cx = centroids.at<double>(i, 0);
                double cy = centroids.at<double>(i, 1);
                circle(dst,  cv::Point(cx,cy),1,cv::Scalar(0, 0, 0),3);
            }
            Moments moment = moments(points);
            mu.push_back(moment);
            vector<double> huMomentsTemp;
            HuMoments(moment, huMomentsTemp);
            for (int j = 0; j < 7; j++){
                huMomentsTemp[j] = -1 * copysign(1.0, huMomentsTemp[j]) * log10(abs(huMomentsTemp[j]));
            }
            huMomentsMain.push_back(huMomentsTemp);
        }
    }
    else{
        std::vector<Vec3b> colors(nLabels);
        colors[0] = Vec3b(0, 0, 0);//background
        int flag = 1;
        for(int label = 1; label < nLabels; ++label){
            if(flag == 1){
                colors[label] = Vec3b( 0, 254, 0);
                flag = 2;
                continue;
            }
            if(flag == 2)
            {
                colors[label] = Vec3b( 255, 0, 0);
                flag =0;
                continue;
            }    
            colors[label] = Vec3b( 0, 0, 255);
            flag = 1;

        }
        Mat dstImg(src.size(), CV_8UC3);
        for(int r = 0; r < dstImg.rows; ++r){
            for(int c = 0; c < dstImg.cols; ++c){
                int label = labelImage.at<int>(r, c);
                Vec3b &pixel = dstImg.at<Vec3b>(r, c);
                pixel = colors[label];
            }
        }
        dst = dstImg.clone();
    }
    return 1;
}



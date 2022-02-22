#include <iostream>
#include <opencv2/opencv.hpp>
#include "csv_util.cpp"
#include <dirent.h>
#include<filter.h>

using namespace cv;
using namespace std;

vector<float> getFeaturesTaskOne(Mat src){
    /*
    Given an image this function will append the integer values of each pixel in the centre 9*9 of the image
    in order of Blue, Green, Red into the 1D vector, so at the end the 1-D vector will have 243 entries (9*9*3)
    */
    std::vector<float> feature;
    int colCount = src.cols/2-4;
    int rowCount = src.rows/2-4;
    for(int x = rowCount; x <= rowCount + 8; x++){
        for(int y = colCount; y <= colCount + 8; y++){
            Vec3b color = src.at<Vec3b>(x,y);
            feature.push_back(color[0]);
            feature.push_back(color[1]);
            feature.push_back(color[2]);
        }
    }
    return feature;
}

vector<float> getFeaturesTaskTwo(Mat src, int bins=8){
    /*
    Given the number of bins this calculates the RGB histogram using bins and at the end normalises the color histogram
    */
    const int Hsize = bins;
    const int divisor = 256/Hsize;
    Mat hist3D;
    int dim3[3] = {Hsize, Hsize, Hsize};
    hist3D = Mat::zeros(3, dim3, CV_32S);
    int count = 0;
    for(int x = 0; x <= src.rows; x++){
        for(int y = 0; y <= src.cols; y++){
            Vec3b color = src.at<Vec3b>(x,y);
            int B = color[0]/divisor;
            int G = color[1]/divisor;
            int R = color[2]/divisor;
            hist3D.at<unsigned int>(B,G,R)++;
            count++;
        }
    }
    vector<float> result;
    for(int i = 0 ; i<Hsize ; i++ ){
        for(int j = 0 ; j<Hsize ; j++ ){
            for(int k = 0 ; k<Hsize ; k++ ){
                result.push_back((hist3D.at<unsigned int>(i,j,k))/float(count));
            }
        }
    }
    return result;
}

vector<float> getFeaturesTaskThree(Mat src, int startRows, int endRows, int startCols, int endCols, int bins=8){
    /*
    Given the number of bins this calculates the RGB histogram using bins and at the end normalises the color histogram
    Adding additional arguments to above function can be combined and when made into library
    The extra arguments are to divide the images and get only histogram of that portion of the image
    */
    const int Hsize = bins;
    const int divisor = 256/Hsize;
    Mat hist3D;
    int dim3[3] = {Hsize, Hsize, Hsize};
    hist3D = Mat::zeros(3, dim3, CV_32S);
    int count = 0;
    for(int x = startRows; x <= endRows; x++){
        for(int y = startCols; y <= endCols; y++){
            Vec3b color = src.at<Vec3b>(x,y);
            int B = color[0]/divisor;
            int G = color[1]/divisor;
            int R = color[2]/divisor;
            hist3D.at<unsigned int>(B,G,R)++;
            count++;
        }
    }
    vector<float> result;
    for(int i = 0 ; i<Hsize ; i++ ){
        for(int j = 0 ; j<Hsize ; j++ ){
            for(int k = 0 ; k<Hsize ; k++ ){
                result.push_back((hist3D.at<unsigned int>(i,j,k))/float(count));
            }
        }
    }
    return result;
}

vector<float> getFeaturesTaskTwo2D(Mat src, int bins=8){
    /*
    Calculates the 2D RG chromaticity histogram using bins 
    */
    const int Hsize = bins;
    const int divisor = 256/Hsize;
    Mat hist2D;
    int dim[2] = {Hsize, Hsize};
    hist2D = Mat::zeros(2, dim, CV_32S);
    int count = 0;
    for(int x = 0; x <= src.rows; x++){
        for(int y = 0; y <= src.cols; y++){
            Vec3b color = src.at<Vec3b>(x,y);
            float temp = color[0] + color[1] + color[2] + 1; // 1 added to remove divide by zero error and it dosent equal to bin in the next step to avoid array out of bound
            int r = (Hsize*color[2])/temp;
            int g = (Hsize*color[1])/temp; // first multiplied by bin size so that the value never becomes 0
            hist2D.at<unsigned int>(r,g)++;
            count++;
        }
    }
    vector<float> result;
    for(int i = 0 ; i<Hsize ; i++ ){
        for(int j = 0 ; j<Hsize ; j++ ){
                result.push_back((hist2D.at<unsigned int>(i,j)/float(count)));
        }
    }
    return result;
}

vector<float> getBlueBinFeatures(Mat src, int bins=8){
    /*
    Given an image this function will append the integer values of each pixel in the centre 9*9 of the image
    in order of Blue, Green, Red into the 1D vector, so at the end the 1-D vector will have 243 entries (9*9*3)
    */
    Mat frame_HSV, frame_threshold;
    //Converting the images and finding the blue objects only
    cvtColor(src, frame_HSV, COLOR_BGR2HSV);
    inRange(frame_HSV, Scalar(100, 150, 0), Scalar(140, 255, 255), frame_threshold);     
    Mat temp, mask;
    cv::cvtColor(frame_threshold, temp, cv::COLOR_GRAY2RGB);
    //Removing all the noises using erode
    erode(temp, mask, getStructuringElement(MORPH_RECT, Size(40, 40)));
    //temp1 will contain only blue objects in white format next we will morph

    Mat maskedImage;
    src.copyTo(maskedImage, mask);


    const int Hsize = bins;
    const int divisor = 256/Hsize;
    Mat hist2D;
    int dim[2] = {Hsize, Hsize};
    hist2D = Mat::zeros(2, dim, CV_32S);
    int count = 0;
    for(int x = 100; x <= maskedImage.rows-100; x++){
        for(int y = 100; y <= maskedImage.cols-100; y++){
            Vec3b color = maskedImage.at<Vec3b>(x,y);
            float temp = color[0] + color[1] + color[2] + 1; // 1 added to remove divide by zero error and it dosent equal to bin in the next step to avoid array out of bound
            int r = (Hsize*color[2])/temp;
            int g = (Hsize*color[1])/temp; // first multiplied by bin size so that the value never becomes 0
            hist2D.at<unsigned int>(r,g)++;
            count++;
        }
    }
    vector<float> result;
    for(int i = 0 ; i<Hsize ; i++ ){
        for(int j = 0 ; j<Hsize ; j++ ){
                result.push_back((hist2D.at<unsigned int>(i,j)/float(count)));
        }
    }
    return result;
}

vector<float> getFeaturesTaskFive(Mat src, int bins = 8){
    /*
    In this function we are finding two features
    i) Cropping the image to centre portion and finding only the red items present there
    ii) Magnitude gradient of the entire image
    This is not the actual task 5 this is an extension
    */
    Mat hsv, threshHSV, white;
    cv::Rect myROI(200, 200, 200, 200);

    cvtColor(src, hsv, COLOR_BGR2HSV);
    inRange(hsv, Scalar(0, 70, 50), Scalar(10, 255, 255), threshHSV);   

    cvtColor(threshHSV, white, COLOR_GRAY2RGB);

    Mat maskedImage;
    medianBlur( white, hsv, 7 );

    src.copyTo(maskedImage, hsv);
    Mat SecondFeature;
    SecondFeature = src.clone();
    Mat croppedImage = maskedImage(myROI);
    float div = 256;
    std::vector<float> feature;
    for(int x = 0; x <= croppedImage.rows; x++){
        for(int y = 0; y <= croppedImage.cols; y++){
            Vec3b color = src.at<Vec3b>(x,y);
            feature.push_back(color[0]/div);
            feature.push_back(color[1]/div);
            feature.push_back(color[2]/div);
        }
    }
    return feature;    
}

int bubbleSortIndex(vector<float> &distanceMetric, vector<int> &imageIndex){
    /*
    Modified bubble sort, where the sorting happens for 5 elements and the bubble sort gets called for N images
    So the complexity is good O(N*5) i.e. O(N), storing the value and the image index
    */
    for(int k = 0; k < distanceMetric.size()-1; k++){
        for(int z = 0 ; z < distanceMetric.size()-k-1; z++){
            if(distanceMetric[z] > distanceMetric[z+1]){
                double temp = distanceMetric[z];
                int tempIndex = imageIndex[z];
                distanceMetric[z] = distanceMetric[z+1];
                imageIndex[z] = imageIndex[z+1];
                distanceMetric[z+1] = temp;
                imageIndex[z+1] = tempIndex;
            }
        }
    }
    return 0;
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
    return sum;
}

float sumOfAbsDiff(vector<float> src, vector<float> dest){
    /*
    Sum of Absolute Difference Distance Metric Calculation
    */
    float sum = 0.0;
    for(int i = 0; i < src.size() ; i++){
        sum += abs(src[i] - dest[i]);
    }
    //cout<<sum;
    return sum;
}

float histogramIntersection(vector<float> src, vector<float> dest){
    /*
    Histogram Intersection distance metric calculation
    */
    float sum = 0.0;
    for(int i = 0; i < src.size() ; i++){
        sum += min(src[i], dest[i]);
    }
    return 1-sum; // subtracting so that if the distance is more it means more dissimilar
}

int readImagesExtractTaskOne(string path, string csvFileName){
    /*
    Function to read images and create the CSV file if the CSV file is not present already for task1
    */
    char dirname[256];
    char filename[256];
    strcpy(dirname, path.c_str());      
    DIR *dirp;
    struct dirent *dp;
    int imgCount = 0;
    int rstFlag = 1;
    dirp = opendir( dirname );
    if( dirp == NULL) {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }
    char csvName[csvFileName.length()];
    strcpy(csvName, csvFileName.c_str());
    while( (dp = readdir(dirp)) != NULL ) {
    // check if the file is an image
        if( strstr(dp->d_name, ".jpg") ||   strstr(dp->d_name, ".png") ||   strstr(dp->d_name, ".ppm") ||   strstr(dp->d_name, ".tif") ) {
            // build the overall filename
            strcpy(filename, dirname);
            strcat(filename, dp->d_name);
            vector<float> resImg = getFeaturesTaskOne(imread(filename, IMREAD_COLOR));
            append_image_data_csv(csvName, filename, resImg, rstFlag);
            imgCount++;
            rstFlag = 0;
        }
    }
    cout<<"Number of images = "<<imgCount<<endl;
    return imgCount;
}

vector<string> readBackImageTaskOne(string imagePath, string path, int topMatchCount){
    /*
    Function that fetches the csv file and given an image compares the feature and returns the top N matches based on the sumOfSquares distance metric
    */
    vector<char *> imageNames;
    topMatchCount++; //Need to change things to remove this extra statement
    std::vector<std::vector<float>> featureVectors;
    char filename[path.length()];
    int i;
    strcpy(filename, path.c_str());    
    char imageName[path.length()];
    strcpy(imageName, imagePath.c_str());
    read_image_data_csv(filename, imageNames, featureVectors, 0);
    for(i = 0;i < imageNames.size(); i++){
        if( strcmp(imageNames[i], imageName) == 0){
            break;
        }
    }
    int count = 0, j=0; //j will contain the index where to begin the rest of the matching and adding it to the top 5 matrix values from 
    vector<int> imageIndex;
    vector<float> distanceMetric;
    while(count < topMatchCount){
        if(j != i){
            imageIndex.push_back(j);
            distanceMetric.push_back(sumOfSquares(featureVectors[i], featureVectors[j]));
            count++;
        }
        j++;
    }
    bubbleSortIndex(distanceMetric, imageIndex);

    for(int k = j; k<imageNames.size(); k++){
        //check if the sum of squares is less than the maximum value
        double sumOfSqua = sumOfSquares(featureVectors[i], featureVectors[k]);
        if(sumOfSqua < distanceMetric[topMatchCount-1] && i != k){
            distanceMetric[topMatchCount-1] = sumOfSqua;
            imageIndex[topMatchCount-1] = k;
            bubbleSortIndex(distanceMetric, imageIndex);
        }
    }
    vector<string> resultImages;
    for(int k = 0; k<topMatchCount-1;k++){
        resultImages.push_back(imageNames[imageIndex[k]]);
    }
    return resultImages;
}

int readImagesExtractTaskTwo(string path, string csvFileName){
    char dirname[256];
    char filename[256];
    strcpy(dirname, path.c_str());      
    DIR *dirp;
    struct dirent *dp;
    int imgCount = 0;
    int rstFlag = 1;
    dirp = opendir( dirname );
    if( dirp == NULL) {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }
    char csvName[csvFileName.length()];
    strcpy(csvName, csvFileName.c_str());
    while( (dp = readdir(dirp)) != NULL ) {
    // check if the file is an image
        if( strstr(dp->d_name, ".jpg") ||   strstr(dp->d_name, ".png") ||   strstr(dp->d_name, ".ppm") ||   strstr(dp->d_name, ".tif") ) {
            // build the overall filename
            strcpy(filename, dirname);
            strcat(filename, dp->d_name);
            vector<float> resImg = getFeaturesTaskTwo(imread(filename, IMREAD_COLOR));
            //vector<float> featureVal(resImg.begin<unsigned int>(), resImg.end<unsigned int>());
            //cout<<sizeof(resImg);
            append_image_data_csv(csvName, filename, resImg, rstFlag);
            imgCount++;
            rstFlag = 0;
        }
    }
    cout<<"Number of images = "<<imgCount<<endl;
    return imgCount;
}

vector<string> readBackImageTaskTwo(string imagePath, string path, int topMatchCount){
    vector<char *> imageNames;
    topMatchCount++; //Need to change things to remove this extra statement
    std::vector<std::vector<float>> featureVectors;
    char filename[path.length()];
    int i;
    strcpy(filename, path.c_str());    
    char imageName[path.length()];
    strcpy(imageName, imagePath.c_str());
    read_image_data_csv(filename, imageNames, featureVectors, 0);
    for(i = 0;i < imageNames.size(); i++){
        if( strcmp(imageNames[i], imageName) == 0){
             cout<<"Found the image "<<i<<endl;
            break;
        }
    }
    //featureVectors[i] will contain the features of the image given
    // select first 5 images different from the input image as the closest
    int count = 0, j=0; //j will contain the index where to begin the rest of the matching and adding it to the top 5 matrix values from 
    vector<int> imageIndex;
    vector<float> distanceMetric;
    while(count < topMatchCount){
        if(j != i){
            imageIndex.push_back(j);
            distanceMetric.push_back(histogramIntersection(featureVectors[i], featureVectors[j]));
            count++;
        }
        j++;
    }
    bubbleSortIndex(distanceMetric, imageIndex);

    for(int k = j; k<imageNames.size(); k++){
        //check if the sum of squares is less than the maximum value
        double sumOfSqua = histogramIntersection(featureVectors[i], featureVectors[k]);
        if(sumOfSqua < distanceMetric[topMatchCount-1] && i != k){
            distanceMetric[topMatchCount-1] = sumOfSqua;
            imageIndex[topMatchCount-1] = k;
            bubbleSortIndex(distanceMetric, imageIndex);
        }
    }
    vector<string> resultImages;
    for(int k = 0; k<topMatchCount-1;k++){
        resultImages.push_back(imageNames[imageIndex[k]]);
    }
    return resultImages;
}

int readImagesExtractTaskThree(string path, string csvFileName){
    char dirname[256];
    char filename[256];
    strcpy(dirname, path.c_str());      
    DIR *dirp;
    struct dirent *dp;
    int imgCount = 0;
    int rstFlag = 1;
    dirp = opendir( dirname );
    if( dirp == NULL) {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }
    char csvName[csvFileName.length()];
    strcpy(csvName, csvFileName.c_str());
    while( (dp = readdir(dirp)) != NULL ) {
    // check if the file is an image
        if( strstr(dp->d_name, ".jpg") ||   strstr(dp->d_name, ".png") ||   strstr(dp->d_name, ".ppm") ||   strstr(dp->d_name, ".tif") ) {
            // build the overall filename
            strcpy(filename, dirname);
            strcat(filename, dp->d_name);
            Mat src = imread(filename, IMREAD_COLOR);
            vector<float> topHalf = getFeaturesTaskThree(src,0, (src.rows)/2,0, src.cols,8);
            vector<float> bottomHalf = getFeaturesTaskThree(src,src.rows/2 + 1, src.rows,0, src.cols,8);

            //vector<float> featureVal(resImg.begin<unsigned int>(), resImg.end<unsigned int>());
            //cout<<sizeof(resImg);
            topHalf.insert(topHalf.end(), bottomHalf.begin(), bottomHalf.end());

            append_image_data_csv(csvName, filename, topHalf, rstFlag);
            imgCount++;
            rstFlag = 0;
        }
    }
    cout<<"Number of images = "<<imgCount<<endl;
    return imgCount;
}

vector<string> readBackImageTaskThree(string imagePath, string path, int topMatchCount){
    vector<char *> imageNames;
    topMatchCount++; //Need to change things to remove this extra statement
    std::vector<std::vector<float>> featureVectors;
    char filename[path.length()];
    int i;
    strcpy(filename, path.c_str());    
    char imageName[path.length()];
    strcpy(imageName, imagePath.c_str());
    read_image_data_csv(filename, imageNames, featureVectors, 0);
    for(i = 0;i < imageNames.size(); i++){
        if( strcmp(imageNames[i], imageName) == 0){
            //cout<<imageName<<endl;
            //cout<<featureVectors[i].size()<<endl;
            cout<<"Found the image "<<i<<endl;
            //cout<<imageNames[i]<<endl;
            break;
        }
    }
    int count = 0, j=0; //j will contain the index where to begin the rest of the matching and adding it to the top 5 matrix values from 
    vector<int> imageIndex;
    vector<float> distanceMetric;
    while(count < topMatchCount){
        if(j != i){
            imageIndex.push_back(j);
            distanceMetric.push_back(histogramIntersection(featureVectors[i], featureVectors[j]));
            count++;
        }
        j++;
    }
    bubbleSortIndex(distanceMetric, imageIndex);

    for(int k = j; k<imageNames.size(); k++){
        //check if the sum of squares is less than the maximum value
        double sumOfSqua = histogramIntersection(featureVectors[i], featureVectors[k]);
        if(sumOfSqua < distanceMetric[topMatchCount-1] && i != k){
            distanceMetric[topMatchCount-1] = sumOfSqua;
            imageIndex[topMatchCount-1] = k;
            bubbleSortIndex(distanceMetric, imageIndex);
        }
    }
    vector<string> resultImages;
    for(int k = 0; k<topMatchCount-1;k++){
        resultImages.push_back(imageNames[imageIndex[k]]);
    }
    return resultImages;
}

int readImagesExtractTaskBlueBin(string path, string csvFileName){
    char dirname[256];
    char filename[256];
    strcpy(dirname, path.c_str());      
    DIR *dirp;
    struct dirent *dp;
    int imgCount = 0;
    int rstFlag = 1;
    dirp = opendir( dirname );
    if( dirp == NULL) {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }
    char csvName[csvFileName.length()];
    strcpy(csvName, csvFileName.c_str());
    while( (dp = readdir(dirp)) != NULL ) {
    // check if the file is an image
        if( strstr(dp->d_name, ".jpg") ||   strstr(dp->d_name, ".png") ||   strstr(dp->d_name, ".ppm") ||   strstr(dp->d_name, ".tif") ) {
            // build the overall filename
            strcpy(filename, dirname);
            strcat(filename, dp->d_name);
            Mat src = imread(filename, IMREAD_COLOR);
            vector<float> resImg = getBlueBinFeatures(src,8);
            //vector<float> featureVal(resImg.begin<unsigned int>(), resImg.end<unsigned int>());
            //cout<<sizeof(resImg);
            Mat xS, yS, mag;
            sobelY3x3(src, xS);
            sobelX3x3(src, yS);
            magnitudeSobel(xS, yS, mag);
                        
            vector<float> textHist = getFeaturesTaskTwo(mag,8);

            //vector<float> featureVal(resImg.begin<unsigned int>(), resImg.end<unsigned int>());
            //cout<<sizeof(resImg);
            resImg.insert(resImg.end(), textHist.begin(), textHist.end());

            append_image_data_csv(csvName, filename, resImg, rstFlag);
            imgCount++;
            rstFlag = 0;
            cout<<"Processing Image = "<<imgCount<<endl;
        }
    }
    cout<<"Number of images = "<<imgCount<<endl;
    return imgCount;
}

vector<string> readBackImageTaskBlueBin(string imagePath, string path, int topMatchCount){
    vector<char *> imageNames;
    topMatchCount++; //Need to change things to remove this extra statement
    std::vector<std::vector<float>> featureVectors;
    char filename[path.length()];
    int i;
    strcpy(filename, path.c_str());    
    char imageName[path.length()];
    strcpy(imageName, imagePath.c_str());
    read_image_data_csv(filename, imageNames, featureVectors, 0);
    for(i = 0;i < imageNames.size(); i++){
        if( strcmp(imageNames[i], imageName) == 0){
            //cout<<imageName<<endl;
            //cout<<featureVectors[i].size()<<endl;
            cout<<"Found the image "<<i<<endl;
            //cout<<imageNames[i]<<endl;
            break;
        }
    }
    //featureVectors[i] will contain the features of the image given
    // select first 5 images different from the input image as the closest
    int count = 0, j=0; //j will contain the index where to begin the rest of the matching and adding it to the top 5 matrix values from 
    vector<int> imageIndex;
    vector<float> distanceMetric;
    while(count < topMatchCount){
        if(j != i){
            imageIndex.push_back(j);
            distanceMetric.push_back(histogramIntersection(featureVectors[i], featureVectors[j]));
            count++;
        }
        j++;
    }
    bubbleSortIndex(distanceMetric, imageIndex);

    for(int k = j; k<imageNames.size(); k++){
        //check if the sum of squares is less than the maximum value
        double sumOfSqua = histogramIntersection(featureVectors[i], featureVectors[k]);
        if(sumOfSqua < distanceMetric[topMatchCount-1] && i != k){
            distanceMetric[topMatchCount-1] = sumOfSqua;
            imageIndex[topMatchCount-1] = k;
            bubbleSortIndex(distanceMetric, imageIndex);
        }
    }
    vector<string> resultImages;
    for(int k = 0; k<topMatchCount-1;k++){
        resultImages.push_back(imageNames[imageIndex[k]]);
    }
    return resultImages;
}

int readImagesExtractTaskTwoExtension(string path, string csvFileName){
    char dirname[256];
    char filename[256];
    strcpy(dirname, path.c_str());      
    DIR *dirp;
    struct dirent *dp;
    int imgCount = 0;
    int rstFlag = 1;
    dirp = opendir( dirname );
    if( dirp == NULL) {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }
    char csvName[csvFileName.length()];
    strcpy(csvName, csvFileName.c_str());
    while( (dp = readdir(dirp)) != NULL ) {
    // check if the file is an image
        if( strstr(dp->d_name, ".jpg") ||   strstr(dp->d_name, ".png") ||   strstr(dp->d_name, ".ppm") ||   strstr(dp->d_name, ".tif") ) {
            // build the overall filename
            strcpy(filename, dirname);
            strcat(filename, dp->d_name);
            vector<float> resImg = getFeaturesTaskTwo2D(imread(filename, IMREAD_COLOR),16);
            append_image_data_csv(csvName, filename, resImg, rstFlag);
            imgCount++;
            rstFlag = 0;
        }
    }
    cout<<"Number of images = "<<imgCount<<endl;
    return imgCount;
}

vector<string> readBackImageTaskTwoExtension(string imagePath, string path, int topMatchCount){
    vector<char *> imageNames;
    topMatchCount++; //Need to change things to remove this extra statement
    std::vector<std::vector<float>> featureVectors;
    char filename[path.length()];
    int i;
    strcpy(filename, path.c_str());    
    char imageName[path.length()];
    strcpy(imageName, imagePath.c_str());
    read_image_data_csv(filename, imageNames, featureVectors, 0);
    for(i = 0;i < imageNames.size(); i++){
        if( strcmp(imageNames[i], imageName) == 0){
            //cout<<imageName<<endl;
            //cout<<featureVectors[i].size()<<endl;
            cout<<"Found the image "<<i<<endl;
            //cout<<imageNames[i]<<endl;
            break;
        }
    }
    //featureVectors[i] will contain the features of the image given
    // select first 5 images different from the input image as the closest
    int count = 0, j=0; //j will contain the index where to begin the rest of the matching and adding it to the top 5 matrix values from 
    vector<int> imageIndex;
    vector<float> distanceMetric;
    while(count < topMatchCount){
        if(j != i){
            imageIndex.push_back(j);
            distanceMetric.push_back(histogramIntersection(featureVectors[i], featureVectors[j]));
            count++;
        }
        j++;
    }
    bubbleSortIndex(distanceMetric, imageIndex);

    for(int k = j; k<imageNames.size(); k++){
        //check if the sum of squares is less than the maximum value
        double sumOfSqua = histogramIntersection(featureVectors[i], featureVectors[k]);
        if(sumOfSqua < distanceMetric[topMatchCount-1] && i != k){
            distanceMetric[topMatchCount-1] = sumOfSqua;
            imageIndex[topMatchCount-1] = k;
            bubbleSortIndex(distanceMetric, imageIndex);
        }
    }
    vector<string> resultImages;
    for(int k = 0; k<topMatchCount-1;k++){
        resultImages.push_back(imageNames[imageIndex[k]]);
    }
    return resultImages;
}

int readImagesExtractTaskFour(string path, string csvFileName){
    char dirname[256];
    char filename[256];
    strcpy(dirname, path.c_str());      
    DIR *dirp;
    struct dirent *dp;
    int imgCount = 0;
    int rstFlag = 1;
    dirp = opendir( dirname );
    if( dirp == NULL) {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }
    char csvName[csvFileName.length()];
    strcpy(csvName, csvFileName.c_str());
    while( (dp = readdir(dirp)) != NULL ) {
    // check if the file is an image
        if( strstr(dp->d_name, ".jpg") ||   strstr(dp->d_name, ".png") ||   strstr(dp->d_name, ".ppm") ||   strstr(dp->d_name, ".tif") ) {
            // build the overall filename
            strcpy(filename, dirname);
            strcat(filename, dp->d_name);
            Mat src = imread(filename, IMREAD_COLOR);
            vector<float> colorHist = getFeaturesTaskTwo(src,8);
            Mat xS, yS, mag;
            sobelY3x3(src, xS);
            sobelX3x3(src, yS);
            magnitudeSobel(xS, yS, mag);
                        
            vector<float> textHist = getFeaturesTaskTwo(mag,8);
            colorHist.insert(colorHist.end(), textHist.begin(), textHist.end());

            append_image_data_csv(csvName, filename, colorHist, rstFlag);
            imgCount++;
            rstFlag = 0;
        }
    }
    cout<<"Number of images = "<<imgCount<<endl;
    return imgCount;
}

vector<string> readBackImageTaskFour(string imagePath, string path, int topMatchCount){
    vector<char *> imageNames;
    topMatchCount++; //Need to change things to remove this extra statement
    std::vector<std::vector<float>> featureVectors;
    char filename[path.length()];
    int i;
    strcpy(filename, path.c_str());    
    char imageName[path.length()];
    strcpy(imageName, imagePath.c_str());
    read_image_data_csv(filename, imageNames, featureVectors, 0);
    for(i = 0;i < imageNames.size(); i++){
        if( strcmp(imageNames[i], imageName) == 0){
            //cout<<imageName<<endl;
            //cout<<featureVectors[i].size()<<endl;
            cout<<"Found the image "<<i<<endl;
            //cout<<imageNames[i]<<endl;
            break;
        }
    }
    //featureVectors[i] will contain the features of the image given
    // select first 5 images different from the input image as the closest
    int count = 0, j=0; //j will contain the index where to begin the rest of the matching and adding it to the top 5 matrix values from 
    vector<int> imageIndex;
    vector<float> distanceMetric;
    while(count < topMatchCount){
        if(j != i){
            imageIndex.push_back(j);
            distanceMetric.push_back(histogramIntersection(featureVectors[i], featureVectors[j]));
            count++;
        }
        j++;
    }
    bubbleSortIndex(distanceMetric, imageIndex);

    for(int k = j; k<imageNames.size(); k++){
        //check if the sum of squares is less than the maximum value
        double sumOfSqua = histogramIntersection(featureVectors[i], featureVectors[k]);
        if(sumOfSqua < distanceMetric[topMatchCount-1] && i != k){
            distanceMetric[topMatchCount-1] = sumOfSqua;
            imageIndex[topMatchCount-1] = k;
            bubbleSortIndex(distanceMetric, imageIndex);
        }
    }
    vector<string> resultImages;
    for(int k = 0; k<topMatchCount-1;k++){
        resultImages.push_back(imageNames[imageIndex[k]]);
    }
    return resultImages;
}

int readImagesExtractTaskFive(string path, string csvFileName){
char dirname[256];
    char filename[256];
    strcpy(dirname, path.c_str());      
    DIR *dirp;
    struct dirent *dp;
    int imgCount = 0;
    int rstFlag = 1;
    dirp = opendir( dirname );
    if( dirp == NULL) {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }
    char csvName[csvFileName.length()];
    strcpy(csvName, csvFileName.c_str());
    while( (dp = readdir(dirp)) != NULL ) {
    // check if the file is an image
        if( strstr(dp->d_name, ".jpg") ||   strstr(dp->d_name, ".png") ||   strstr(dp->d_name, ".ppm") ||   strstr(dp->d_name, ".tif") ) {
            // build the overall filename
            strcpy(filename, dirname);
            strcat(filename, dp->d_name);
            Mat src = imread(filename, IMREAD_COLOR);
            cv::Rect myROI(100, 100, 400, 400);
            Mat croppedImage = src(myROI);

            Mat frame_HSV, frame_threshold;
            //Converting the images and finding the blue objects only
            cvtColor(croppedImage, frame_HSV, COLOR_BGR2HSV);


            Mat mask1, mask2;
            inRange(frame_HSV, Scalar(0, 70, 50), Scalar(10, 255, 255), mask1);
            inRange(frame_HSV, Scalar(170, 70, 50), Scalar(180, 255, 255), mask2);

            Mat mask = mask1 | mask2;

            cv::cvtColor(mask, frame_threshold, cv::COLOR_GRAY2RGB);
            //Removing all the noises using erode
            //temp1 will contain only blue objects in white format next we will morph

            Mat maskedImage;
            croppedImage.copyTo(maskedImage, frame_threshold);


            vector<float> colorHist = getFeaturesTaskTwo2D(maskedImage,8);
            Mat xS, yS, mag;
            sobelY3x3(croppedImage, xS);
            sobelX3x3(croppedImage, yS);
            magnitudeSobel(xS, yS, mag);
                        
            vector<float> textHist = getFeaturesTaskTwo(mag,8);

            //vector<float> featureVal(resImg.begin<unsigned int>(), resImg.end<unsigned int>());

            colorHist.insert(colorHist.end(), textHist.begin(), textHist.end());

            append_image_data_csv(csvName, filename, colorHist, rstFlag);
            imgCount++;

            rstFlag = 0;
        }
    }
    cout<<"Number of images = "<<imgCount<<endl;
    return imgCount;}

vector<string> readBackImageTaskFive(string imagePath, string path, int topMatchCount){
    vector<char *> imageNames;
    topMatchCount++; //Need to change things to remove this extra statement
    std::vector<std::vector<float>> featureVectors;
    char filename[path.length()];
    int i;
    strcpy(filename, path.c_str());    
    char imageName[path.length()];
    strcpy(imageName, imagePath.c_str());
    read_image_data_csv(filename, imageNames, featureVectors, 0);
    for(i = 0;i < imageNames.size(); i++){
        if( strcmp(imageNames[i], imageName) == 0){
            cout<<"Found the image "<<i<<endl;
            break;
        }
    }
    int count = 0, j=0; //j will contain the index where to begin the rest of the matching and adding it to the top 5 matrix values from 
    vector<int> imageIndex;
    vector<float> distanceMetric;
    while(count < topMatchCount){
        if(j != i){
            imageIndex.push_back(j);
            float sumColorHist = histogramIntersection({featureVectors[i].begin(), featureVectors[i].begin() + 255}, {featureVectors[j].begin(), featureVectors[j].begin() + 255});
            float sumtextHist = histogramIntersection({featureVectors[i].begin()+256, featureVectors[i].end()}, {featureVectors[j].begin()+256, featureVectors[j].end()});

            distanceMetric.push_back(0.1*sumtextHist + 0.75*sumColorHist);
            count++;
        }
        j++;
    }
    bubbleSortIndex(distanceMetric, imageIndex);

    for(int k = j; k<imageNames.size(); k++){
        //check if the sum of squares is less than the maximum value
        float sumColorHist = histogramIntersection({featureVectors[i].begin(), featureVectors[i].begin() + 255}, {featureVectors[j].begin(), featureVectors[j].begin() + 255});
        float sumtextHist = histogramIntersection({featureVectors[i].begin()+256, featureVectors[i].end()}, {featureVectors[j].begin()+256, featureVectors[j].end()});
        float sumOfSqua = 0.1*sumtextHist + 0.75*sumColorHist;
        if(sumOfSqua < distanceMetric[topMatchCount-1] && i != k){
            distanceMetric[topMatchCount-1] = sumOfSqua;
            imageIndex[topMatchCount-1] = k;
            bubbleSortIndex(distanceMetric, imageIndex);
        }
    }
    vector<string> resultImages;
    for(int k = 0; k<topMatchCount-1;k++){
        resultImages.push_back(imageNames[imageIndex[k]]);
        cout<<distanceMetric[imageIndex[k]];
    }
    return resultImages;
}

int readImagesExtractTaskFourExtension(string path, string csvFileName){
    /*
    Using the Laws filter along with the 3D color histogram with same weight
    */
    char dirname[256];
    char filename[256];
    strcpy(dirname, path.c_str());      
    DIR *dirp;
    struct dirent *dp;
    int imgCount = 0;
    int rstFlag = 1;
    dirp = opendir( dirname );
    if( dirp == NULL) {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }
    char csvName[csvFileName.length()];
    strcpy(csvName, csvFileName.c_str());
    while( (dp = readdir(dirp)) != NULL ) {
    // check if the file is an image
        if( strstr(dp->d_name, ".jpg") ||   strstr(dp->d_name, ".png") ||   strstr(dp->d_name, ".ppm") ||   strstr(dp->d_name, ".tif") ) {
            // build the overall filename
            strcpy(filename, dirname);
            strcat(filename, dp->d_name);
            Mat src = imread(filename, IMREAD_COLOR);
            vector<float> colorHist = getFeaturesTaskTwo(src,8);
            Mat mag;
            lawsFilterE5L5(src, mag);
            vector<float> textHist = getFeaturesTaskTwo(mag,8);
            colorHist.insert(colorHist.end(), textHist.begin(), textHist.end());
            append_image_data_csv(csvName, filename, colorHist, rstFlag);
            imgCount++;
            rstFlag = 0;
        }
    }
    cout<<"Number of images = "<<imgCount<<endl;
    return imgCount;
}

vector<string> readBackImageTaskFourExtension(string imagePath, string path, int topMatchCount){
    vector<char *> imageNames;
    topMatchCount++; //Need to change things to remove this extra statement
    std::vector<std::vector<float>> featureVectors;
    char filename[path.length()];
    int i;
    strcpy(filename, path.c_str());    
    char imageName[path.length()];
    strcpy(imageName, imagePath.c_str());
    read_image_data_csv(filename, imageNames, featureVectors, 0);
    for(i = 0;i < imageNames.size(); i++){
        if( strcmp(imageNames[i], imageName) == 0){
            cout<<"Found the image "<<i<<endl;
            break;
        }
    }
    int count = 0, j=0; //j will contain the index where to begin the rest of the matching and adding it to the top 5 matrix values from 
    vector<int> imageIndex;
    vector<float> distanceMetric;
    while(count < topMatchCount){
        if(j != i){
            imageIndex.push_back(j);
            distanceMetric.push_back(histogramIntersection(featureVectors[i], featureVectors[j]));
            count++;
        }
        j++;
    }
    bubbleSortIndex(distanceMetric, imageIndex);

    for(int k = j; k<imageNames.size(); k++){
        //check if the sum of squares is less than the maximum value
        double sumOfSqua = histogramIntersection(featureVectors[i], featureVectors[k]);
        if(sumOfSqua < distanceMetric[topMatchCount-1] && i != k){
            distanceMetric[topMatchCount-1] = sumOfSqua;
            imageIndex[topMatchCount-1] = k;
            bubbleSortIndex(distanceMetric, imageIndex);
        }
    }
    vector<string> resultImages;
    for(int k = 0; k<topMatchCount-1;k++){
        resultImages.push_back(imageNames[imageIndex[k]]);
    }
    return resultImages;
}

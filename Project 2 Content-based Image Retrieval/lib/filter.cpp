#include <filter.h>
using namespace std;

int lawsFilterE5L5( Mat &src, Mat &dst ){
    float image_data[25] = {-1, -4, -6, -4, -1, -2, -8, -12, -8, -2, 0,0,0,0,0, 2,8,12,8,2,1,4,6,4,1};
    Mat image = cv::Mat(5, 5, CV_32F, image_data);
    cv::filter2D(src, dst, -1, image, cv::Point(-1, -1), 0,cv::BORDER_DEFAULT);
    return 0;
}

int sobelX3x3( Mat &src, Mat &dst ){
    /*
    Takes input image src and then using convolution generates and stores xSobel in dst
    */
    int sobelX1[3][1] = {1,2,1};
    Mat temp;
    temp.create(src.size(),CV_16SC3);
    dst.create(temp.size(),CV_16SC3);
    for(int x = 1; x < src.rows-1; x++){
        for(int y = 1; y < src.cols-1; y++){
            for(int c=0 ; c<3 ; c++){
                int magX = 0;
                int picMat[3][1] = {src.at<Vec3b>(x-1,y)[c], src.at<Vec3b>(x,y)[c], src.at<Vec3b>(x+1,y)[c]};
                for(int i=0 ; i<3 ; i++){
                        magX = magX + sobelX1[i][0] * picMat[i][0];                      
                }
                temp.at<Vec3s>(x,y)[c] = magX/4;
            }
        }
    }
    int sobelX2[1][3] = {-1,0,1};
    for(int x = 1; x < src.rows-1; x++){
        for(int y = 1; y < src.cols-1; y++){
            for(int c=0 ; c<3 ; c++){
                int magX = 0;
                int picMat[1][3] = {src.at<Vec3b>(x,y-1)[c], src.at<Vec3b>(x,y)[c], src.at<Vec3b>(x,y+1)[c]};
                for(int i=0 ; i<3 ; i++){
                        magX = magX + sobelX2[0][i] * picMat[0][i];                        
                }
                dst.at<Vec3s>(x,y)[c] = abs(magX);
            }
        }
    }
    return 0;
}

int sobelY3x3( Mat &src, Mat &dst ){
    /*
        Takes input image src and then using convolution generates and stores ySobel in dst

    */
    int sobelY1[3][1] = {1,0,-1};
    Mat temp;
    temp.create(src.size(),CV_16SC3);
    dst.create(temp.size(),CV_16SC3);
    for(int x = 1; x < src.rows-1; x++){
        for(int y = 1; y < src.cols-1; y++){
            for(int c=0 ; c<3 ; c++){
                int magX = 0;
                int picMat[3][1] = {src.at<Vec3b>(x-1,y)[c], src.at<Vec3b>(x,y)[c], src.at<Vec3b>(x+1,y)[c]};
                for(int i=0 ; i<3 ; i++){
                        magX = magX + sobelY1[i][0] * picMat[i][0];                      
                }
                temp.at<Vec3s>(x,y)[c] = magX;
            }
        }
    }
    dst = temp.clone();
    return 0;
    int sobelY2[1][3] = {1,2,1};
    for(int x = 1; x < src.rows-1; x++){
        for(int y = 1; y < src.cols-1; y++){
            for(int c=0 ; c<3 ; c++){
                int magX = 0;
                int picMat[1][3] = {src.at<Vec3b>(x,y-1)[c], src.at<Vec3b>(x,y)[c], src.at<Vec3b>(x,y+1)[c]};
                for(int i=0 ; i<3 ; i++){
                        magX = magX + sobelY2[0][i] * picMat[0][i];                        
                }
                dst.at<Vec3s>(x,y)[c] = (magX/4);
            }
        }
    }
    return 0;
}

int magnitudeSobel( Mat &sx, Mat &sy, Mat &dst ){ 
    /*
    Generates the Gradient Magnitude using xSobel and ySobel
    */
    dst.create(sx.size(),CV_8UC3);
    Mat temp;
    temp.create(sx.size(),CV_16SC3);
    for(int x = 0; x < sx.rows; x++){
        for(int y = 0; y < sx.cols; y++){
            for(int c=0 ; c<3 ; c++){
                temp.at<Vec3s>(x,y)[c] = sqrt(sx.at<Vec3s>(x,y)[c] * sx.at<Vec3s>(x,y)[c] + sy.at<Vec3s>(x,y)[c] * sy.at<Vec3s>(x,y)[c]);     //how to use -255 to 255
            }
        }
    }
    convertScaleAbs(temp, dst);
    return 0;
}

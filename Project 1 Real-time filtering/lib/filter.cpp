#include<includeFile.h>
#include<filter.h>
using namespace std;

int greyScale(Mat &src, Mat &dst){
    dst = src.clone();
    for(int x = 0; x < src.rows; x++){
        for(int y = 0; y < src.cols; y++){
            Vec3b color = src.at<Vec3b>(x,y);
            uchar avg = (uchar) (0.0722 * color.val[0] + 0.7152 * color.val[1] + 0.2126 * color.val[2]);
            Vec3b destColor = dst.at<Vec3b>(x,y);
            destColor.val[0] = destColor.val[1] = destColor.val[2] = color.val[0];
            dst.at<Vec3b>(x,y) = destColor;
        }
    }
    return 0;
}

int blur5x5(Mat &src, Mat &dst){
    
    Mat temp = src.clone();
    int blurFilter[] = {1, 2, 4, 2, 1};
    
    for(int x = 0; x < src.rows; x++){
        for(int y = 2; y < src.cols-2; y++){                //Removing 2 from sides as they are corner cases

            int result[] = {0,0,0};
            for(int j = y-2; j <= y+2 ; j++){
                Vec3b intensity = src.at<Vec3b>(x,j);
                int filter_P = blurFilter[j - (y-2)];
                for(int i = 0; i<3 ; i++){
                    result[i] += intensity[i] * filter_P;
                }
            }

            for(int i = 0; i<3 ; i++){
                temp.at<Vec3b>(x,y)[i] = (int) result[i] / 10;
            }
        }
    }
    dst = temp.clone();
    //Applying the second filter
    for(int x = 2; x < temp.rows - 2; x++){
        for(int y = 0; y < temp.cols; y++){                //Removing 2 from sides as they are corner cases

            int result[] = {0,0,0};
            for(int j = x-2; j <= x+2 ; j++){

                Vec3b intensity = temp.at<Vec3b>(j,y);
                int filter_P = blurFilter[j - (x-2)];
                for(int i = 0; i<3 ; i++){
                    result[i] += intensity[i] * filter_P;
                }
            }
            for(int i = 0; i<3 ; i++){
                dst.at<Vec3b>(x,y)[i] = (int) result[i] / 10;
            }
        }
    }
    return 0;
}

int sobelX3x3NonSeparable( Mat &src, Mat &dst ){
    int sobelX1[3][1] = {1,2,1};
    Mat temp;
    temp.create(src.size(),CV_8SC3);
    dst.create(temp.size(),CV_8SC3);
    for(int x = 1; x < src.rows-1; x++){
        for(int y = 1; y < src.cols-1; y++){
            for(int c=0 ; c<3 ; c++){
                int magX = 0;
                int picMat[3][1] = {src.at<Vec3b>(x-1,y)[c], src.at<Vec3b>(x,y)[c], src.at<Vec3b>(x+1,y)[c]};
                for(int i=0 ; i<3 ; i++){
                        magX = magX + sobelX1[i][0] * picMat[i][0];                      
                }
                temp.at<Vec3b>(x,y)[c] = magX;     //how to use -255 to 255
            }
        }
    }
    //return 0;
    int sobelX2[1][3] = {-1,0,1};
    for(int x = 1; x < src.rows-1; x++){
        for(int y = 1; y < src.cols-1; y++){
            for(int c=0 ; c<3 ; c++){
                int magX = 0;
                int picMat[1][3] = {src.at<Vec3b>(x,y-1)[c], src.at<Vec3b>(x,y)[c], src.at<Vec3b>(x,y+1)[c]};
                for(int i=0 ; i<3 ; i++){
                        magX = magX + sobelX2[0][i] * picMat[0][i];                        
                }
                dst.at<Vec3b>(x,y)[c] = (magX);     //how to use -255 to 255
            }
        }
    }
    //dst = temp.clone();
    return 0;
}
   
int sobelY3x3nonSeparable( Mat &src, Mat &dst ){
    int sobelMatrix[3][3] = {1,2,1, 0,0,0,-1,-2,-1};
    //int sobelMatrix[3][3] = {-1,-2,-1, 0,0,0,1,2,1};
    dst = src.clone();
    for(int x = 1; x < src.rows-1; x++){
        for(int y = 1; y < src.cols-1; y++){
            for(int c=0 ; c<3 ; c++){
                int magX = 0;
                int picMat[3][3] = {src.at<Vec3b>(x-1,y-1)[c], src.at<Vec3b>(x-1,y)[c], src.at<Vec3b>(x-1,y+1)[c], src.at<Vec3b>(x,y-1)[c], src.at<Vec3b>(x,y)[c], src.at<Vec3b>(x,y+1)[c],src.at<Vec3b>(x+1,y-1)[c], src.at<Vec3b>(x+1,y)[c], src.at<Vec3b>(x+1,y+1)[c]};
                for(int i=0 ; i<3 ; i++){
                    for(int j=0 ; j<3 ; j++){
                        magX = magX + sobelMatrix[i][j] * picMat[i][j];
                    }
                }
                if(magX > 255){
                    magX = 255;
                }
                /*else if(magX< -255){
                    magX = -255;
                }*/
                dst.at<Vec3b>(x,y)[c] = magX/4;
            }
        }
    }
    return 0;
}

int sobelX3x3( Mat &src, Mat &dst ){
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

int magnitudeSobel( Mat &sx, Mat &sy, Mat &dst ){ //Need to update
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

int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels ){
    Mat tmp;
    blur5x5(src, tmp);
    int bucketSize = 255/levels;
    dst = tmp.clone();
    for(int x = 0; x < tmp.rows; x++){
        for(int y = 0; y < tmp.cols; y++){
            for(int c=0 ; c<3 ; c++){
                dst.at<Vec3b>(x,y)[c] = (int(tmp.at<Vec3b>(x,y)[c]))/ bucketSize * bucketSize ;
            }
        }
    }
    return 0;

}

int cartoon( cv::Mat &src, cv::Mat&dst, int levels, int magThreshold ){
    Mat xSobelFrame, ySobelFrame, sobelFrame, blurFrame, xSob, ySob;
    sobelX3x3(src, xSobelFrame);
    sobelY3x3(src, ySobelFrame);
    magnitudeSobel(xSobelFrame, ySobelFrame, sobelFrame);
    blurQuantize(src, blurFrame, levels);
    
    dst = blurFrame.clone();
    for(int x = 0; x < src.rows; x++){
        for(int y = 0; y < src.cols; y++){
            for(int c=0 ; c<3 ; c++){
                if(sobelFrame.at<Vec3b>(x,y)[c] > magThreshold ){
                    dst.at<Vec3b>(x,y)[c] = 0;
                }
            }
        }
    }
    return 0;
}

int brightnessIncrease(Mat &src, Mat &dst, int brightness){
    dst = src.clone();
    for(int x = 0; x < src.rows; x++){
        for(int y = 0; y < src.cols; y++){
            for(int c=0 ; c<3 ; c++){
                dst.at<Vec3b>(x,y)[c] = saturate_cast< uchar >(src.at<Vec3b>(x,y)[c] + brightness);
            }
        }
    }
    return 0;
}

int laplaceFilter(Mat &src, Mat &dst){
    dst = src.clone();
    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            int sum = src.at<uchar>(y - 1, x)
                + src.at<uchar>(y + 1, x)
                + src.at<uchar>(y, x - 1)
                + src.at<uchar>(y, x + 1)
                - 4 * src.at<uchar>(y, x);

            dst.at<uchar>(y, x) = cv::saturate_cast<uchar>(sum);
        }
    }
    return 0;
}

int putLegendText( cv::Mat &src, cv::Mat &dst, std::vector<String> stats){
    dst = src.clone();
    for( int i=0; i < stats.size(); i++){
        putText(dst, stats[i].c_str(), Point2f(src.cols-300,(src.rows-700)+(i*20)), FONT_HERSHEY_PLAIN, 1.2, Scalar(255,0,0,0), 1);
    }
    return 0;
}

int sharpeness( cv::Mat &src, cv::Mat &dst){
    dst.create(src.size(),CV_16SC3);
    Mat temp;
    temp.create(src.size(),CV_16SC3);
    for(int x = 1; x < src.rows-1; x++){
        for(int y = 1; y < src.cols-1; y++){
            for(int c=0 ; c<3 ; c++){
                int magX = 0;
                magX = src.at<Vec3b>(x,y)[c] *  5 + (-1) * src.at<Vec3b>(x - 1,y)[c] + (-1) * src.at<Vec3b>(x,y-1)[c] + (-1) * src.at<Vec3b>(x +1,y)[c] + (-1) * src.at<Vec3b>(x,y+1)[c];
                temp.at<Vec3s>(x,y)[c] = magX;
            }
        }
    }
    convertScaleAbs(temp, dst);
    return 0;

}

int reSize( cv::Mat &src, cv::Mat &dst, float dx, float dy){
    resize(src, dst, Size(), dx, dy);
    return 0;
}

int colorPalleteChange(cv::Mat &src, cv::Mat &dst){
    dst = src.clone();
    int bucketSize = 15;
    int levels = 10;
    Mat xSobelFrame, ySobelFrame, sobelFrame, blurFrame, xSob, ySob;
    sobelX3x3(src, xSobelFrame);

    sobelY3x3(src, ySobelFrame);
    magnitudeSobel(xSobelFrame, ySobelFrame, sobelFrame);
    blurQuantize(src, blurFrame, levels);

    for(int x = 0; x < src.rows; x++){
        for(int y = 0; y < src.cols; y++){
            dst.at<Vec3b>(x,y)[0] = blurFrame.at<Vec3b>(x,y)[1];
            dst.at<Vec3b>(x,y)[1] = blurFrame.at<Vec3b>(x,y)[2];
            dst.at<Vec3b>(x,y)[2] = blurFrame.at<Vec3b>(x,y)[0];
            for(int c=0 ; c<3 ; c++){
                if(sobelFrame.at<Vec3b>(x,y)[c] > 15 ){
                    dst.at<Vec3b>(x,y)[c] = 0;
                }
            }
        }
    }
    return 0;
}

int medianFilter(cv::Mat &src, cv::Mat &dst){
    dst = src.clone();

    for(int x = 1; x < src.rows-1; x++){
        for(int y = 1; y < src.cols-1; y++){
            for(int c=0 ; c<3 ; c++){
                int magX = 0;
                int picMat[] = {src.at<Vec3b>(x-1,y-1)[c], src.at<Vec3b>(x-1,y)[c], src.at<Vec3b>(x-1,y+1)[c], src.at<Vec3b>(x,y-1)[c], src.at<Vec3b>(x,y)[c], src.at<Vec3b>(x,y+1)[c],src.at<Vec3b>(x+1,y-1)[c], src.at<Vec3b>(x+1,y)[c], src.at<Vec3b>(x+1,y+1)[c]};
                std::sort(picMat,picMat+9);
                dst.at<Vec3b>(x,y)[c] = picMat[8];
            }
        }
    }
    return 0;
}
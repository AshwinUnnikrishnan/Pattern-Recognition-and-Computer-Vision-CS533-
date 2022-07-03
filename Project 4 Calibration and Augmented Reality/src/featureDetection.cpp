#include<filter.h>
#include <GL/glew.h>

#include <GLFW/glfw3.h>

using namespace cv;
using namespace std;
//https://cpp.hotexamples.com/examples/-/-/calibrateCamera/cpp-calibratecamera-function-examples.html
int main(int argc, char const *argv[])
{
        Size patternSize(9,6);
        VideoCapture *capdev;
        capdev = new VideoCapture(1);
        if( !capdev->isOpened() ) {
                printf("Unable to open video device\n");
                return(-1);
        }
        Size refS( (int) capdev->get(CAP_PROP_FRAME_WIDTH ),(int) capdev->get(CAP_PROP_FRAME_HEIGHT));
        namedWindow("Video", 1); // identifies a window
        Mat colorFrame, res;
        char key2 = 'n';
        while(1) {
            *capdev >> colorFrame; // get a new frame from the camera, treat as a stream
            if( colorFrame.empty() ) {
                printf("frame is empty\n");
                break;
            }
            char key;
            switch( key = waitKey(1) ){ // Find feature points on the input format
                case 'h':
                    key2 = 'h';
                    break;
                case 's':
                    key2 = 's';
                    break;
                case 'q':
                    return(0);
                    break;
                case 'S':
                    key2 = 'S';
                    break;
                case 'n':
                    key2 = 'n';
                    break;
                default:
                    break;
            }
            switch( key2 ){ // Find feature points on the input format
                case 'h':
                    cornerHarris_d(colorFrame, res);
                    break;
                case 's':
                    myShiTomasi_function(colorFrame, res);
                    break;
                case 'S':
                    cornerHarris_d(colorFrame, res);
                    imwrite("Harris.jpg", res); // A JPG FILE IS BEING SAVED
                    imwrite("Harris_Original.jpg", colorFrame); // A JPG FILE IS BEING SAVED
                    key2 = 'h';
                    break;
                case 'q':
                    return(0);
                    break;
                default:
                    res = colorFrame.clone();
                    break;
            }
            cv::imshow("Video", res);
        }
        delete capdev;
        return(0);
}


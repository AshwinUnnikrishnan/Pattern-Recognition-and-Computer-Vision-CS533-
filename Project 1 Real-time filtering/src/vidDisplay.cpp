#include<includeFile.h>
#include<filter.h>
using namespace cv;
using namespace std;

int main(int argc, char const *argv[])
{
        VideoCapture *capdev;
        capdev = new VideoCapture(0);           //Opening Camera
        if( !capdev->isOpened() ) {
                printf("Unable to open video device\n");
                return(-1);
        }
        Size refS( (int) capdev->get(CAP_PROP_FRAME_WIDTH ),(int) capdev->get(CAP_PROP_FRAME_HEIGHT));
        Mat colorFrame, blurFrame, currentFrame, xSobelFrame, ySobelFrame, sobelFrame, tempFrame, textFrame, prevFrame, testF, testFcolor;
        int countImage = 0, countVideo = 0, flag = 0, brightMax = 100, brightMin = -100, brightness = 0;
        String onScreenText = "";
        VideoWriter video;
        String tempText;
        namedWindow("Video", 1); // identifies a window
        std::vector<String> legend{"g : OpenCV Greyscale","h : Own Greyscale","x : Sobel X","y : Sobel Y","m : Gradient Magnitude","c : Cartoonizing","+ : Increase Brightness","- : Decrease Brightness","z : default testing line","b : Gaussian Blur","v : Start and Stop Recoring","s : Save Current Frame","i : Blur and Quantize","! : Laplace Filter","w : Negative Inverted","n : Set back to normal","f : Ghost Effect","a : Sharpen Frames","r : Resize","p : Change Color Palette","u : Median Filter","q : Exit"};
        
        while(1) {
                *capdev >> colorFrame; // get a new frame from the camera, treat as a stream
                currentFrame = colorFrame;
                if( colorFrame.empty() ) {
                  printf("frame is empty\n");
                  break;
                }
                switch(char key = waitKey(1)){
                    case 'q':
                        return(0);
                    case 'v':
                        if(onScreenText == ""){
                            onScreenText = "Recording Video press v to stop";
                            countVideo++;
                            cout<<"Enter the text to save in the Video as a meme\n";
                            cin>>tempText;
                            video = VideoWriter("videoCapture_Count" + to_string(countVideo)+ ".avi", VideoWriter::fourcc('M','J','P','G'), 20, Size(refS.width, refS.height));
                        }
                        else{
                            onScreenText = "";
                            video.release();
                        }
                        break;
                    case 's':
                        countImage++;
                        cout<<"Enter the text to save in the Image as a meme\n";
                        cin>>tempText;
                        putText(testF, tempText,Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 2, LINE_4);
                        imwrite("ImageCapture_Count" + to_string(countImage)+ ".jpg", testF); // A JPG FILE IS BEING SAVED
                        imwrite("ImageCapture_CountOriginal" + to_string(countImage)+ ".jpg", testFcolor); // A JPG FILE IS BEING SAVED
                        break;
                    case 'g':
                        flag = 1;
                        break;
                    case 'h':
                        flag = 2;
                        break;
                    case 'b':
                        flag = 3;
                        break;
                    case 'n':
                        brightness = 0;
                        flag = 4;
                        break;
                    case 'x':
                        flag = 5;
                        break;
                    case 'y':
                        flag = 6;
                        break;
                    case 'm':
                        flag = 7;
                        break;
                    case 'i':
                        flag = 8;
                        break;
                    case '+':
                        brightness++;
                        flag = 9;
                        break;
                    case '-':
                        brightness--;
                        flag = 10;
                        break;
                    case '!':
                        flag = 11;
                        break;
                    case 'c':
                        flag = 12;
                        break;
                    case 'w': //negative
                        flag = 13;
                        break;
                    case 'f':
                        flag = 14;
                        break;
                    case 'a':
                        flag = 15;
                        break;
                    case 'r':
                        flag = 16;
                        break;
                    case 'p':
                        flag = 17;
                        break;
                    case 'u':
                        flag = 18;
                        break;
                    case 'z':
                        flag = 20;
                        break;
                }
                testFcolor = colorFrame.clone();
                switch(flag){
                    case 1:
                        cvtColor(colorFrame, currentFrame, COLOR_BGR2GRAY);
                        break;
                    case 2:
                        greyScale(colorFrame, currentFrame);
                        break;
                    case 3:
                        blur5x5(colorFrame, currentFrame);
                        break;
                    case 5:
                        sobelX3x3(colorFrame, xSobelFrame);
                        convertScaleAbs(xSobelFrame, currentFrame);
                        break;
                    case 6:
                        sobelY3x3(colorFrame, ySobelFrame);
                        convertScaleAbs(ySobelFrame, currentFrame);
                        break;
                    case 7:
                        sobelY3x3(colorFrame, ySobelFrame);
                        sobelX3x3(colorFrame, xSobelFrame);
                        magnitudeSobel(xSobelFrame, ySobelFrame, currentFrame);
                        break;
                    case 8:
                        blurQuantize(colorFrame, currentFrame, 15);
                        break;
                    case 9:
                        if(brightness > 100)
                            brightness = 100;
                        brightnessIncrease(colorFrame, currentFrame, brightness);
                        break;
                    case 10:
                        if(brightness < -100)
                            brightness = -100;
                        brightnessIncrease(colorFrame, currentFrame, brightness);
                        break;
                    case 11:
                        laplaceFilter(colorFrame, currentFrame);
                        break;
                    case 12:
                        cartoon(colorFrame, currentFrame, 15, 15);
                        break;
                    case 13:
                        cvtColor(colorFrame, blurFrame,COLOR_BGR2GRAY);//Converting BGR to Grayscale image and storing it into 'converted' matrix//
                        cv::threshold(blurFrame, sobelFrame, 100, 255, THRESH_BINARY);//converting grayscale image stored in 'converted' matrix into binary image//
                        cv::bitwise_not(sobelFrame, currentFrame);//inverting the binary image and storing it in inverted_binary_image matrix//
                        break;
                    case 14:
                        addWeighted(colorFrame, 0.3, prevFrame, 0.7, 0.0, currentFrame);
                        break;
                    case 15:
                        sharpeness(colorFrame, currentFrame);
                        break;                    
                    case 16:
                        reSize(colorFrame, currentFrame);
                        break;
                    case 17:
                        colorPalleteChange(colorFrame, currentFrame);
                        break;
                    case 18:
                        medianFilter(colorFrame, currentFrame);
                        break;                    
                }
                prevFrame = currentFrame.clone();
                if(onScreenText.compare("Recording Video press v to stop") == 0){
                    tempFrame = currentFrame.clone();
                    putText(tempFrame, tempText,Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2, LINE_4);
                    video.write(tempFrame);//Saving Video
                    //when recording showing that it is recording
                    putText(currentFrame, onScreenText,Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2, LINE_4);
                }
                //Putting list of operations available
                putLegendText(currentFrame, textFrame, legend);
                cv::imshow("Video", textFrame);
                testF = currentFrame.clone();
        }
        delete capdev;
        return(0);
}


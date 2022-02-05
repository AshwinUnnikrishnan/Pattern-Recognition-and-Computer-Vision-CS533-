#include<includeFile.h>
#include<filter.h>

using namespace cv;
using namespace std;

int main(int argc, char const *argv[])
{
    string image_path = samples::findFile("tempDog.jpg",false);
    Mat colorFrame = imread(image_path, IMREAD_COLOR);
    Mat updFrame, xS, yS, temp, currentFrame, textFrame;

    if(colorFrame.empty()){
        cout << "Could not read the image: " << image_path << std::endl;
        return 1;   
    }
    cout<<"Displaying the image\n";
    namedWindow("PhotoFrame",WINDOW_AUTOSIZE); // identifies a window
    
    updFrame = colorFrame.clone();
    currentFrame = colorFrame.clone();
    temp = colorFrame.clone();
    //putText(colorFrame, "Press Any Key For Edit Options", Point2f(200,300), FONT_HERSHEY_SCRIPT_COMPLEX, 1.5, Scalar(255,0,0,0), 1);
    //imshow("PhotoFrame", colorFrame);
    std::vector<String> legend{"g : OpenCV Greyscale","h : Own Greyscale","x : Sobel X","y : Sobel Y","m : Gradient Magnitude","c : Cartoonizing","+ : Increase Brightness","- : Decrease Brightness","b : Gaussian Blue","s : Save Current Frame","i : Blur and Quantize","! : Laplace Filter","w : Negative Inverted","n : Seet back to normal","a : Sharpen Frames","r : Resize","p : Change Color Palatte","u : Median Filter","q : Exit"};
    int inp = waitKey(0); // Wait for a keystroke in the window
    int flag = 0; // flag = 0 means color 1 means grey version
    int brightMax = 100, brightMin = -100, brightness = 0;
    int countImage = 0;
    while(1) {
                if( colorFrame.empty() ) {
                  printf("frame is empty\n");
                  break;
                }
                switch(char key = waitKey(1)){
                    case 'q':
                        return(0);
                    case 's':
                        cout<<"DSF";
                        imwrite("SavePhoto_"+to_string(countImage)+".jpg", updFrame); // A JPG FILE IS BEING SAVED
                        countImage++;
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
                        flag = 4;
                        brightness = 0;
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
                    case 'w': 
                        flag = 13;
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
                }
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
                    case 4:
                        currentFrame = temp.clone();
                        break;
                    case 5:
                        sobelX3x3(colorFrame, updFrame);
                        convertScaleAbs(updFrame, currentFrame);
                        break;
                    case 6:
                        sobelY3x3(colorFrame, updFrame);
                        convertScaleAbs(updFrame, currentFrame);
                        break;
                    case 7:
                        sobelY3x3(colorFrame, xS);
                        sobelX3x3(colorFrame, yS);
                        magnitudeSobel(xS, yS, currentFrame);
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
                        cartoon(colorFrame, currentFrame, 15, 20);
                        break;
                    case 13:
                        cvtColor(colorFrame, xS,COLOR_BGR2GRAY);//Converting BGR to Grayscale image and storing it into 'converted' matrix//
                        cv::threshold(xS, updFrame, 100, 255, THRESH_BINARY);//converting grayscale image stored in 'converted' matrix into binary image//
                        cv::bitwise_not(updFrame, currentFrame);//inverting the binary image and storing it in inverted_binary_image matrix//
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
                putLegendText(currentFrame, textFrame, legend);
                imshow("PhotoFrame", textFrame);
                updFrame = currentFrame.clone();
        }
    return 0;
}



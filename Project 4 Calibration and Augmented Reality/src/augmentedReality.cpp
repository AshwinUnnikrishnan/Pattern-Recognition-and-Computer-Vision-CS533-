#include<filter.h>
//#include<GLFW/glfw3.h>
using namespace cv;
using namespace std;
//https://cpp.hotexamples.com/examples/-/-/calibrateCamera/cpp-calibratecamera-function-examples.html

String calibStr(bool cal, int chessCount){
    String calib = "Not yet calibrated";
    if(cal == true){
        calib = "Calibrated";
    }
    else{
        calib = "Not enough images for calibration";
        if(chessCount >= 5){
            calib = "Ready for calibration enough images collected";
        }
    }
    return calib;
}

int main(int argc, char const *argv[])
{
        Size patternSize(9,6);
        Size patternSize_1(4,11);
        VideoCapture *capdev;
        capdev = new VideoCapture(0);
        if( !capdev->isOpened() ) {
                printf("Unable to open video device\n");
                return(-1);
        }
        Size refS( (int) capdev->get(CAP_PROP_FRAME_WIDTH ),(int) capdev->get(CAP_PROP_FRAME_HEIGHT));
        Mat cameraMat;
        bool drawRect = false;
        bool hideBackground = false;
        buildCameraMatrix(float(refS.width)/2, float(refS.height)/2, cameraMat);
        Mat distCoeffs;
        int temp = 0;
        distCoeffs = Mat::zeros(5, 1, CV_64F);
        std::vector<Mat> rvecs, tvecs;
        bool cal = false;
        bool liveChess = true;                       // so that till the point user dosent click any input it detects and shows chessboard corners if it exists
        bool positionCal = false;
        Mat colorFrame, prevChessBoardInstance;
        namedWindow("Video", 1); // identifies a window
        namedWindow("ChessBoard", 1); // identifies a window
        int chessCount = 0;
        std::vector<cv::Point2f> corner_set;
        std::vector<cv::Point2f> circle_corner_set;
        std::vector<std::vector<cv::Point2f> > corner_list;
        std::vector<std::vector<cv::Point2f> > circle_corner_list;
        std::vector<std::vector<cv::Vec3f> > point_list;
        std::vector<std::vector<cv::Vec3f> > circle_point_list;
        double rms;
        while(1) {
            *capdev >> colorFrame; // get a new frame from the camera, treat as a stream
            if( colorFrame.empty() ) {
                printf("frame is empty\n");
                break;
            }
            bool found, foundCircle;
            int chessBoardFlags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK;

            char key;
            switch( key = waitKey(1) ){ // Find feature points on the input format
                case 's':
                    liveChess = false;              //Stop drawing chess corners in the live video and then only show the objects or what tasks asks
                    found = findChessboardCorners( colorFrame, patternSize, corner_set, chessBoardFlags);
                    break;
                case 'c':               //calibrate
                    if(chessCount <5)
                        cout<<"Not Enough images to calibrate"<<endl; // Can add it to screen as calibrating
                    else{              
                        printMat(cameraMat, "Camera Matrix Before calibration");
                        printMat(distCoeffs, "Distortion Coefficient Before calibration");
                        rms = calibrateCamera(point_list, corner_list, refS, cameraMat, distCoeffs, rvecs, tvecs, CALIB_FIX_ASPECT_RATIO);                        
                        printMat(cameraMat, "Camera Matrix After calibration");   
                        printMat(distCoeffs, "Distortion Coefficient After calibration");   
                        cout<<"RMS error reported by calibrateCamera: "<<rms<<endl;
                        writeParam("IPad", cameraMat, distCoeffs);
                        cal = true;
                        positionCal = true;
                    }
                    break;
                case 'p':
                    found = findChessboardCorners( colorFrame, patternSize, corner_set, chessBoardFlags);
                    liveChess = true;
                    positionCal = true;
                    break;
                case 'd':
                    liveChess = true;
                    drawRect = !drawRect;
                    positionCal = true;
                    found = findChessboardCorners( colorFrame, patternSize, corner_set, chessBoardFlags);
                    break;
                case 'q':
                    return(0);
                    break;
                case 'C':               // Use CSV to load the camera calibration if exists
                    checkandLoadCalib("CameraConfig.csv","LaptopCamera",cameraMat, distCoeffs);
                    cal = true;
                    positionCal = true;
                    break;
                case 'r':
                    hideBackground = !hideBackground;
                    break;
                case 'i':{
                    string image_path = samples::findFile("chessBoard1.jpg",false);
                    colorFrame = cv::imread(image_path,IMREAD_COLOR);
                    found = findChessboardCorners( colorFrame, patternSize, corner_set, chessBoardFlags);
                    break;
                }
                case 'n':
                    liveChess = true;
                    positionCal = false;            //no break after this so that we set live chess as true and continue drawing chess options
                default:
                    if(liveChess == true)
                        found = findChessboardCorners( colorFrame, patternSize, corner_set, chessBoardFlags);
                        foundCircle = findCirclesGrid( colorFrame, patternSize_1, circle_corner_set, CALIB_CB_ASYMMETRIC_GRID );
                    break;
            }
            
            String calib = calibStr(cal, chessCount);
            if(found == true){                  // Found some feature from above 3 , Check which it belongs to
                if(key == 's'){                 // Chess feature
                    Mat viewGray;
                    chessCount++;
                    cvtColor(colorFrame, viewGray, COLOR_BGR2GRAY);
                    cornerSubPix( viewGray, corner_set, Size(11,11),Size(-1,-1), TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 30, 0.0001 ));
                    prevChessBoardInstance = colorFrame.clone();
                    drawChessboardCorners( prevChessBoardInstance, patternSize, Mat(corner_set), found );
                    std::vector<cv::Vec3f> point_set;
                    for(int i = 0; i < patternSize.height ; i++){
                        for(int j = 0; j < patternSize.width ; j++){
                            Point3f Point3DWorld(j, -i, 0);
                            point_set.push_back(Point3DWorld);
                        }
                    }
                    point_list.push_back(point_set);
                    corner_list.push_back(corner_set);
                    putText(prevChessBoardInstance, "Number of Corners = " + to_string(corner_set.size()), Point2f(50,700), FONT_HERSHEY_PLAIN, 1.2, Scalar(255,0,0,0), 1);
                    putText(prevChessBoardInstance, "Top Left = (" + to_string(corner_set[0].x)+","+ to_string(corner_set[0].y) +")", Point2f(50,650), FONT_HERSHEY_PLAIN, 1.2, Scalar(255,0,0,0), 1);
                    String calibrated = "Cailbrating"; 
                    if(chessCount >= 5)
                        calibrated = "Calibrated";
                    putText(prevChessBoardInstance, calibrated , Point2f(50,600), FONT_HERSHEY_PLAIN, 1.2, Scalar(255,255,0,0), 1);
                    cv::imshow("ChessBoard", prevChessBoardInstance );
                    saveImageAndFeature(colorFrame, prevChessBoardInstance, corner_set, point_set, "chessBoard", chessCount);
                }
                else if(liveChess == true || key == 'i'){
                    Mat viewGray;
                    cvtColor(colorFrame, viewGray, COLOR_BGR2GRAY);
                    cornerSubPix( viewGray, corner_set, Size(11,11),Size(-1,-1), TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 30, 0.0001 ));
                    if(positionCal == true){        //Calculate pos and print on the live video
                        //calculating the 3D points 
                        std::vector<cv::Vec3f> point_set;
                        std::vector<cv::Vec3f> point_set_2;
                        for(int i = 0; i < patternSize.height ; i++){
                            for(int j = 0; j < patternSize.width ; j++){
                                Point3f Point3DWorld(j, -i, 0);
                                point_set.push_back(Point3f(j,-i,0));
                                point_set_2.push_back(Point3f(j, -i, 2));
                            }
                        }
                        Mat rvec1, tvec1, viewT;
                        //undistort(colorFrame, viewT, cameraMat, distCoeffs);
                        //colorFrame = viewT.clone();
                        solvePnP(point_set, corner_set, cameraMat, distCoeffs, rvec1, tvec1);
                        String rotationV = printMat(rvec1, "Rotation Data");
                        String translationV = printMat(tvec1, "Translation Data");
                        putText(colorFrame, "Rotation " + rotationV, Point2f(50,600), FONT_HERSHEY_PLAIN, 1.2, Scalar(255,0,0,0), 1);
                        putText(colorFrame, "Translation " + translationV, Point2f(50,650), FONT_HERSHEY_PLAIN, 1.2, Scalar(255,0,255,0), 1);
                        std::vector<cv::Point2f> corners;
                        projectPoints(point_set, rvec1, tvec1, cameraMat, distCoeffs, corners); //gets all the corners in 3D

                        if(drawRect == false && key != 'i'){
                            drawAxes(colorFrame, rvec1, tvec1, cameraMat, distCoeffs, corner_set[0] );
                            //Need to change this wrt size of chess board and probably make a function out of it
                            circle( colorFrame, corners[0], 10.0, Scalar( 255, 0, 255 ), 1, 8 );
                            circle( colorFrame, corners[8], 10.0, Scalar( 255, 0, 255 ), 1, 8 );
                            circle( colorFrame, corners[45], 10.0, Scalar( 255, 0, 255 ), 1, 8 );
                            circle( colorFrame, corners[53], 10.0, Scalar( 255, 0, 255 ), 1, 8 );    
                        }
                        else{
                            std::vector<cv::Point2f> corners_2;
                            projectPoints(point_set_2, rvec1, tvec1, cameraMat, distCoeffs, corners_2);
                            if(hideBackground == true){
                                hideBackgroundFunc(colorFrame);
                            }
                            drawRectangle(colorFrame, corners, corners_2);
                            if(key == 'i'){
                                imwrite("staticImage1.jpg", colorFrame); // A JPG FILE IS BEING SAVED
                            }
                        }
    
                    }
                    else{                           //Detecting and 
                        drawChessboardCorners( colorFrame, Size(9,6), Mat(corner_set), found );
                        putText(colorFrame, "LiveVideo Number of Corners = " + to_string(corner_set.size()), Point2f(50,700), FONT_HERSHEY_PLAIN, 1.2, Scalar(255,0,0,0), 1);
                        putText(colorFrame, "Top Left = (" + to_string(corner_set[0].x)+","+ to_string(corner_set[0].y) +")", Point2f(50,650), FONT_HERSHEY_PLAIN, 1.2, Scalar(255,0,0,0), 1);
                    }
                }
                //found = false;                  //Setting found=false so that it does not keep searching without user typing 'c' is redundant as already set as false if not rectangle
            }
            if(foundCircle == true){                  // Found some feature from above 3 , Check which it belongs to
                cout<<"Found circle"<<endl;
                imwrite("circleIamge.jpg", colorFrame); // A JPG FILE IS BEING SAVED
                prevChessBoardInstance = colorFrame.clone();
                drawChessboardCorners( prevChessBoardInstance, patternSize_1, Mat(circle_corner_set), foundCircle );
                imwrite("ccccircleIamge.jpg", prevChessBoardInstance); // A JPG FILE IS BEING SAVED
                colorFrame = prevChessBoardInstance.clone();
            }
            putText(colorFrame, calib , Point2f(50,50), FONT_HERSHEY_PLAIN, 1.2, Scalar(255,255,0,0), 1);
                    
            cv::imshow("Video", colorFrame);
        }
        delete capdev;
        return(0);
}


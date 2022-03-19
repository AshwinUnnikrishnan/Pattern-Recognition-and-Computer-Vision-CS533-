#include <wx/filepicker.h>
#include <wx/sysopt.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include "csv_util.cpp"
#include <string>
#include <filter.h>
#include <wx/wxprec.h>
#include <fstream>
#ifndef WX_PRECOMP
#include <wx/wx.h>
#endif

using namespace cv;
using namespace std;



class MyApp : public wxApp
{
public:
    virtual bool OnInit();
};
wxIMPLEMENT_APP(MyApp);

class MyFrame : public wxFrame
{
public:
    wxTextCtrl* LabelInput;
    wxTextCtrl* LabelInput1;
    wxTextCtrl* LabelInput2;
    wxTextCtrl* LabelInput3;
    MyFrame(const wxString &title, const wxPoint &pos, const wxSize &size);
};

bool MyApp::OnInit()
{
    MyFrame *frame = new MyFrame("Project 3", wxDefaultPosition, wxDefaultSize);
    frame->Show(true);
    return true;
}
MyFrame::MyFrame(const wxString &title, const wxPoint &pos, const wxSize &size) : wxFrame(nullptr, wxID_ANY, title, pos, size)
{

    //Creating the two Panels one for buttons to choose different tasks and the other for showing the result
    wxPanel *panel_top = new wxPanel(this, wxID_ANY, wxDefaultPosition, wxSize(200, 100));
    panel_top->SetBackgroundColour(wxColor(100, 100, 200));
    wxPanel *panel_bottom = new wxPanel(this, wxID_ANY, wxDefaultPosition, wxSize(200, 100));
    panel_bottom->SetBackgroundColour(wxColor(100, 200, 100));
    wxBoxSizer *sizer = new wxBoxSizer(wxVERTICAL);
    sizer->Add(panel_top, 1, wxEXPAND | wxLEFT | wxTOP | wxRIGHT, 10);
    sizer->Add(panel_bottom, 5, wxEXPAND | wxALL, 10);
    wxStaticBitmap* selectedImage = new wxStaticBitmap(panel_bottom, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxDefaultSize);

    //wxSizer *sizer_bottom = new wxBoxSizer(wxVERTICAL);
    //panel_bottom->SetSizerAndFit(sizer_bottom);
    this->SetSizerAndFit(sizer);

    wxFlexGridSizer* flexGridSizer = new wxFlexGridSizer(2, 8, 0, 0);     
	panel_top->SetSizer(flexGridSizer);     
    wxFlexGridSizer* flexGridSizer1 = new wxFlexGridSizer(4, 2, 0, 0);     
	panel_bottom->SetSizer(flexGridSizer1);     
	
	wxButton* Task1 = new wxButton(panel_top, wxID_ANY, wxT("Task 1 Binary Threshold"));     
	flexGridSizer->Add(Task1, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);     
	wxButton* Task2 = new wxButton(panel_top, wxID_ANY, wxT("Task 2"));
	flexGridSizer->Add(Task2, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);     
	wxButton* Task3 = new wxButton(panel_top, wxID_ANY, wxT("Task 3"));     
	flexGridSizer->Add(Task3, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5); 
    wxButton* Task4 = new wxButton(panel_top, wxID_ANY, wxT("Task 4"));     
	flexGridSizer->Add(Task4, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5); 
    wxButton* task5 = new wxButton(panel_top, wxID_ANY, wxT("Task 5 Training"));     
	flexGridSizer->Add(task5, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);     
	wxButton* task6 = new wxButton(panel_top, wxID_ANY, wxT("Task 6 Detect"));     
	flexGridSizer->Add(task6, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);     
	
    wxButton* Task7 = new wxButton(panel_top, wxID_ANY, wxT("Task 7 KNN"));     
	flexGridSizer->Add(Task7, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);     
    wxButton* Task8 = new wxButton(panel_top, wxID_ANY, wxT("Task 8 Testing"));     
	flexGridSizer->Add(Task8, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);     
    
    wxButton* Task7_1 = new wxButton(panel_top, wxID_ANY, wxT("Extension KNN Image One Known or Unknown"));     
	flexGridSizer->Add(Task7_1, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);     
	
	wxButton* Task1_1 = new wxButton(panel_top, wxID_ANY, wxT("Task 1 Sat"));     
	flexGridSizer->Add(Task1_1, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);     
	wxButton* Task1_2 = new wxButton(panel_top, wxID_ANY, wxT("Task 1 Self "));     
	flexGridSizer->Add(Task1_2, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);     
	wxButton* def = new wxButton(panel_top, wxID_ANY, wxT("Six Default Check"));     
	flexGridSizer->Add(def, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);
    


    wxStaticText* label = new wxStaticText(panel_bottom, wxID_ANY, "Training Label Current");
    flexGridSizer1->Add(label, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);     

    LabelInput = new wxTextCtrl(panel_bottom, wxID_ANY, "", {10, 10});
    flexGridSizer1->Add(LabelInput, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);     

    wxStaticText* label1 = new wxStaticText(panel_bottom, wxID_ANY, "Image KNN Check");
    flexGridSizer1->Add(label1, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);     

    LabelInput1 = new wxTextCtrl(panel_bottom, wxID_ANY, "", {10, 10});
    flexGridSizer1->Add(LabelInput1, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);     
    
    wxStaticText* label2 = new wxStaticText(panel_bottom, wxID_ANY, "Test Data Folder");
    flexGridSizer1->Add(label2, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);     

    LabelInput2 = new wxTextCtrl(panel_bottom, wxID_ANY, "", {10, 10});
    flexGridSizer1->Add(LabelInput2, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);     

    wxStaticText* label3 = new wxStaticText(panel_bottom, wxID_ANY, "Accuracy");
    flexGridSizer1->Add(label3, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);     

    LabelInput3 = new wxTextCtrl(panel_bottom, wxID_ANY, "", {10, 10});
    flexGridSizer1->Add(LabelInput3, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);     




    
    Task1->Bind(wxEVT_BUTTON, [&](wxCommandEvent& event) {
        /*
            Implemented 4 Types of Thresholding 1 of the 4 used showing the otsu in results
        */
        Mat Fram, res, colorFrame;
        VideoCapture *capdev;
        capdev = new VideoCapture(0);           //Opening Camera
        if( !capdev->isOpened() ) {
                printf("Unable to open video device\n");
                return(-1);
        }
        Size refS( (int) capdev->get(CAP_PROP_FRAME_WIDTH ),(int) capdev->get(CAP_PROP_FRAME_HEIGHT));
        namedWindow("Thresholding", 1); // identifies a window
        char c;
        int imageCount = 1;
        while(1){
            double thresh = 0;
            double maxValue = 255;
            *capdev >> colorFrame; // get a new frame from the camera, treat as a stream
            if( colorFrame.empty() ) {
                  printf("frame is empty\n");
                  break;
            }
            Mat res1, res2, res3, res4;
            thresholdBinary(colorFrame, res, 4);
            thresholdBinary(colorFrame, res1, 1);
            thresholdBinary(colorFrame, res2, 2);
            thresholdBinary(colorFrame, res3, 3);

            imshow("Thresholding", res);
            if((c = waitKey(1)) == 'q'){
                break;
            }
            if(c == 's'){
                //All together saved in the task4 not used here
                imwrite("Threshold_OTSU"+to_string(imageCount)+".jpg", res); // A JPG FILE IS BEING SAVED
                imwrite("Threshold_GreyScaleMethod"+to_string(imageCount)+".jpg", res1); // A JPG FILE IS BEING SAVED
                imwrite("Threshold_HSVMethod"+to_string(imageCount)+".jpg", res2); // A JPG FILE IS BEING SAVED
                imwrite("Threshold_OwnMethod"+to_string(imageCount)+".jpg", res3); // A JPG FILE IS BEING SAVED
                imwrite("OriginalBinary_"+to_string(imageCount)+".jpg", colorFrame); // A JPG FILE IS BEING SAVED
                imageCount += 1;
            }
        }
        delete capdev;
        destroyWindow("Thresholding");
        return 0;
    });
    Task1_1->Bind(wxEVT_BUTTON, [&](wxCommandEvent& event) {
        /*
            Function to do HSV Thresholding
        */
        Mat Fram, res, colorFrame;
        VideoCapture *capdev;
        capdev = new VideoCapture(0);           //Opening Camera
        if( !capdev->isOpened() ) {
                printf("Unable to open video device\n");
                return(-1);
        }
        Size refS( (int) capdev->get(CAP_PROP_FRAME_WIDTH ),(int) capdev->get(CAP_PROP_FRAME_HEIGHT));
        cout<<"ASD"<<endl;
        namedWindow("Thresholding1", 2); // identifies a window
        char c;
        int imageCount = 1;
        while(1){
            *capdev >> colorFrame; // get a new frame from the camera, treat as a stream
            if( colorFrame.empty() ) {
                  printf("frame is empty\n");
                  break;
            }

            thresholdBinary(colorFrame, res, 4);
            imshow("Thresholding1", res);
            if((c = waitKey(1)) == 'q'){
                break;
            }
            if(c == 's'){
                imwrite("HSVThreshold_"+to_string(imageCount)+".jpg", res); // A JPG FILE IS BEING SAVED
                imwrite("OriginalImageHSV_"+to_string(imageCount)+".jpg", colorFrame); // A JPG FILE IS BEING SAVED
                imageCount += 1;
            }
            
        }
        delete capdev;
        destroyWindow("Thresholding1");
        return 0;
    });
    Task1_2->Bind(wxEVT_BUTTON, [&](wxCommandEvent& event) {
        /*
            Function to do HSV Thresholding Testing Function
        */
        Mat Fram, res, colorFrame;
        VideoCapture *capdev;
        capdev = new VideoCapture(0);           //Opening Camera
        if( !capdev->isOpened() ) {
                printf("Unable to open video device\n");
                return(-1);
        }
        Size refS( (int) capdev->get(CAP_PROP_FRAME_WIDTH ),(int) capdev->get(CAP_PROP_FRAME_HEIGHT));
        namedWindow("Thresholding", 1); // identifies a window
        char c;
        int imageCount = 1;
        while(1){
            *capdev >> colorFrame; // get a new frame from the camera, treat as a stream
            if( colorFrame.empty() ) {
                  printf("frame is empty\n");
                  break;
            }
            thresholdBinary(colorFrame, res, 1);
            imshow("Thresholding", res);
            if((c = waitKey(1)) == 'q'){
                break;
            }
            if(c == 's'){
                imwrite("BinaryThreshold_"+to_string(imageCount)+".jpg", res); // A JPG FILE IS BEING SAVED
                imwrite("OriginalBinary_"+to_string(imageCount)+".jpg", colorFrame); // A JPG FILE IS BEING SAVED
                imageCount += 1;
            }
        }
        delete capdev;
        destroyWindow("Thresholding");
        return 0;
    });
    Task2->Bind(wxEVT_BUTTON, [&](wxCommandEvent& event) {
        /*
            Calling the function to perform the cleanup the body patches missed using grow and shrink operations
            Shows the thresholded and the image after growAndShrink
            Performing Shrink initially to remove patches and then grow followed by shrink, total shrink and grow count is same to get back to same size
        */
        Mat Fram, res, colorFrame;
        VideoCapture *capdev;
        capdev = new VideoCapture(0);           //Opening Camera
        if( !capdev->isOpened() ) {
                printf("Unable to open video device\n");
                return(-1);
        }
        Size refS( (int) capdev->get(CAP_PROP_FRAME_WIDTH ),(int) capdev->get(CAP_PROP_FRAME_HEIGHT));
        namedWindow("Thresholded", 1); // identifies a window
        namedWindow("AfterGrowAndShrink", 1); // identifies a window

        char c;
        int imageCount = 1;
        while(1){
            *capdev >> colorFrame; // get a new frame from the camera, treat as a stream
            if( colorFrame.empty() ) {
                  printf("frame is empty\n");
                  break;
            }
            Mat cleanUp, res1, res2, res3, res4;
            thresholdBinary(colorFrame, cleanUp, 4);
            //After Thresholding it cleans up the image using shrink and grow operations
            //Did Shrink 2 iterations to remove the salt pepper small noises and then 7 iterations of grow followed by 5 iterations of shrink to get original shape ratio back
            grow_shrink(cleanUp, res4, 2, 3, 1);
            grow_shrink(res4, res, 7, 3, 0);
            grow_shrink(res, res3, 5, 3, 1);
            imshow("AfterGrowAndShrink", res3);
            imshow("Thresholded", cleanUp);

            if((c = waitKey(1)) == 'q'){
                break;
            }
            if(c == 's'){
                //notSaved here as task 4 takes cummulatively this is for testing only
                imwrite("OriginalColorTask2"+to_string(imageCount)+".jpg", colorFrame); // A JPG FILE IS BEING SAVED
                imwrite("ThresholdedTask2"+to_string(imageCount)+".jpg", cleanUp); // A JPG FILE IS BEING SAVED
                imwrite("Task2AfterGrowing"+to_string(imageCount)+".jpg", res); // A JPG FILE IS BEING SAVED
                imwrite("Tas2AfterGrowAndShrink"+to_string(imageCount)+".jpg", res3); // A JPG FILE IS BEING SAVED
                imageCount += 1;
            }
        }
        delete capdev;
        destroyWindow("Thresholded");
        destroyWindow("AfterGrowAndShrink"); // identifies a window
        return 0;
    });
    Task3->Bind(wxEVT_BUTTON, [&](wxCommandEvent& event) {
        /*
            Segmenting the area into regions to identify the objects
            Outputs the frame as colored regions
        */
        Mat Fram, res, colorFrame;
        VideoCapture *capdev;
        capdev = new VideoCapture(0);           //Opening Camera
        if( !capdev->isOpened() ) {
                printf("Unable to open video device\n");
                return(-1);
        }
        Size refS( (int) capdev->get(CAP_PROP_FRAME_WIDTH ),(int) capdev->get(CAP_PROP_FRAME_HEIGHT));
        namedWindow("Labelled", 1); // identifies a window

        char c;
        int imageCount = 1;
        while(1){
            *capdev >> colorFrame; // get a new frame from the camera, treat as a stream
            if( colorFrame.empty() ) {
                  printf("frame is empty\n");
                  break;
            }
            Mat cleanUp, res1, res2, res3, res4;
            
            thresholdBinary(colorFrame, cleanUp, 4);
            grow_shrink(cleanUp, res4, 2, 3, 1);
            grow_shrink(res4, res, 12, 3, 0);
            grow_shrink(res, res3, 10, 3, 1);
            vector<vector<double>> huMoments;

            conectedCompo(res3, res2, 1, huMoments);

            
            imshow("Labelled", res2);

            if((c = waitKey(1)) == 'q'){
                break;
            }
            if(c == 's'){
                //only for testing purpose
                imwrite("OriginalColorTask2"+to_string(imageCount)+".jpg", colorFrame); // A JPG FILE IS BEING SAVED
                imwrite("ThresholdedTask2"+to_string(imageCount)+".jpg", cleanUp); // A JPG FILE IS BEING SAVED
                imwrite("Labelled"+to_string(imageCount)+".jpg", res); // A JPG FILE IS BEING SAVED
                imwrite("Tas2AfterGrowAndShrink"+to_string(imageCount)+".jpg", res3); // A JPG FILE IS BEING SAVED
                imageCount += 1;
            }
        }
        delete capdev;
        destroyWindow("Labelled");
        return 0;
    });

    Task4->Bind(wxEVT_BUTTON, [&](wxCommandEvent& event) {
        /*
            Calling the componentAnalysiswithStats to return the humoments and the boundedbox Image
            Also calls the regionCover componentAnalysis from Task 3 to show the images comparision
        */
        Mat Fram, res, colorFrame;
        VideoCapture *capdev;
        capdev = new VideoCapture(0);           //Opening Camera
        if( !capdev->isOpened() ) {
                printf("Unable to open video device\n");
                return(-1);
        }
        Size refS( (int) capdev->get(CAP_PROP_FRAME_WIDTH ),(int) capdev->get(CAP_PROP_FRAME_HEIGHT));
        namedWindow("Bounding", 1); // identifies a window

        char c;
        int imageCount = 1;
        while(1){
            *capdev >> colorFrame; // get a new frame from the camera, treat as a stream
            if( colorFrame.empty() ) {
                  printf("frame is empty\n");
                  break;
            }
            Mat cleanUp, res1, res2, res3, res4;
            
            thresholdBinary(colorFrame, cleanUp, 4);
            grow_shrink(cleanUp, res4, 2, 3, 1);
            grow_shrink(res4, res, 12, 3, 0);
            grow_shrink(res, res3, 10, 3, 1);
            vector<vector<double>> huMoments;
            vector<vector<double>> huMoments1;
            conectedCompo(res3, res4, 1, huMoments1);

            conectedCompo(res3, res2, 0, huMoments);

            
            imshow("Bounding", res2);

            if((c = waitKey(1)) == 'q'){
                break;
            }
            if(c == 's'){
                imwrite("Original"+to_string(imageCount)+".jpg", colorFrame); // A JPG FILE IS BEING SAVED
                imwrite("Thresholded_Task1"+to_string(imageCount)+".jpg", cleanUp); // A JPG FILE IS BEING SAVED
                imwrite("Cleanup_Task2"+to_string(imageCount)+".jpg", res3); // A JPG FILE IS BEING SAVED
                imwrite("Labelled_Task3"+to_string(imageCount)+".jpg", res4); // A JPG FILE IS BEING SAVED
                imwrite("Box_Task4"+to_string(imageCount)+".jpg", res2); // A JPG FILE IS BEING SAVED
                imageCount += 1;
            }
        }
        delete capdev;
        destroyWindow("Bounding");
        return 0;
    });

    def->Bind(wxEVT_BUTTON, [&](wxCommandEvent& event) {
        /*
            Calling the function to perform the cleanup the body patches missed
        */
        Mat Fram, res, colorFrame;
        //VideoCapture *capdev;
        //capdev = new VideoCapture(0);           //Opening Camera
        //if( !capdev->isOpened() ) {
        //        printf("Unable to open video device\n");
        //        return(-1);
        //}
        //Size refS( (int) capdev->get(CAP_PROP_FRAME_WIDTH ),(int) capdev->get(CAP_PROP_FRAME_HEIGHT));
        string image_path = samples::findFile("OriginalBinary.jpg",false);
        colorFrame = imread(image_path, IMREAD_COLOR);

        namedWindow("Thresholded", 1); // identifies a window
        namedWindow("AfterGrow", 1); // identifies a window
        namedWindow("AfterGrowAndShrink", 1); // identifies a window

        char c;
        int imageCount = 1;
        while(1){
            //*capdev >> colorFrame; // get a new frame from the camera, treat as a stream
            if( colorFrame.empty() ) {
                  printf("frame is empty\n");
                  break;
            }
            Mat cleanUp, res1, res2, res3, res4;
            thresholdBinary(colorFrame, cleanUp, 1);
            grow_shrink(cleanUp, res4, 2, 3, 1);

            grow_shrink(res4, res, 12, 3, 0);
            grow_shrink(res, res3, 10, 3, 1);
            vector<vector<double>> huMoments;

            conectedCompo(res3, res2, 0, huMoments);
            for(int m=0; m < huMoments.size() ; m++){

                std::vector<float> featureVec(huMoments[m].begin(), huMoments[m].end());
                                
                char *csvName = (char *)"training.csv";
                char *Label = (char *)"Label";

                append_image_data_csv(csvName, Label, featureVec, 0);
                for (int j = 0; j < 7; j++){
                    cout<<huMoments[m][j]<<"     ";
                }
                cout<<endl;
            }
            imshow("AfterGrow", res);
            imshow("AfterGrowAndShrink", res2);
            imshow("Thresholded", cleanUp);

            if((c = waitKey(1)) == 'q'){
                break;
            }
            if(c == 's'){
                imwrite("OriginalColorTask2"+to_string(imageCount)+".jpg", colorFrame); // A JPG FILE IS BEING SAVED
                imwrite("ThresholdedTask2"+to_string(imageCount)+".jpg", cleanUp); // A JPG FILE IS BEING SAVED
                imwrite("Task2AfterGrowing"+to_string(imageCount)+".jpg", res); // A JPG FILE IS BEING SAVED
                imwrite("Tas2AfterGrowAndShrink"+to_string(imageCount)+".jpg", res3); // A JPG FILE IS BEING SAVED
                imageCount += 1;
            }
        }
        //delete capdev;
        destroyWindow("AfterGrow");
        destroyWindow("Thresholded");
        destroyWindow("AfterGrowAndShrink"); // identifies a window
        return 0;
    });

    
    task5->Bind(wxEVT_BUTTON, [&](wxCommandEvent& event) {
        /*
            Need to set the label to the name of the current object getting trained
            this button then stores the features returned from the connectedComponentAnalysis with the label given in the feature vector log normalization is used
        */
        Mat Fram, res, colorFrame;
        VideoCapture *capdev;
        capdev = new VideoCapture(0);           //Opening Camera
        if( !capdev->isOpened() ) {
                printf("Unable to open video device\n");
                return(-1);
        }
        Size refS( (int) capdev->get(CAP_PROP_FRAME_WIDTH ),(int) capdev->get(CAP_PROP_FRAME_HEIGHT));

        namedWindow("TrainingData", 1); // identifies a window
        
        char c;
        int imageCount = 1;
        while(1){
            *capdev >> colorFrame; // get a new frame from the camera, treat as a stream
            if( colorFrame.empty() ) {
                  printf("frame is empty\n");
                  break;
            }
            Mat cleanUp, res1, res2, res3, res4;
            thresholdBinary(colorFrame, cleanUp, 4);
            grow_shrink(cleanUp, res4, 2, 3, 1);

            grow_shrink(res4, res, 12, 3, 0);
            grow_shrink(res, res3, 10, 3, 1);
            vector<vector<double>> huMoments;
            conectedCompo(res3, res2, 0, huMoments);
            int exitFlag = 0;
            switch(c = waitKey(1)){
                case 'q':
                            exitFlag = 1;
                            break;
                case 's':   //not important here testing only
                            imwrite("OriginalColorTask2"+to_string(imageCount)+".jpg", colorFrame); // A JPG FILE IS BEING SAVED
                            imwrite("ThresholdedTask2"+to_string(imageCount)+".jpg", cleanUp); // A JPG FILE IS BEING SAVED
                            imwrite("Task2AfterGrowing"+to_string(imageCount)+".jpg", res); // A JPG FILE IS BEING SAVED
                            imwrite("Tas2AfterGrowAndShrink"+to_string(imageCount)+".jpg", res3); // A JPG FILE IS BEING SAVED
                            imageCount += 1;
                            break;
                case 'n':
                            string labelName = string(LabelInput->GetValue().mb_str());
                            char arr[labelName.length() + 1]; 
                            strcpy(arr, labelName.c_str()); 
                            imwrite("TrainingData_"+labelName+"_"+to_string(imageCount)+".jpg", colorFrame); // A JPG FILE IS BEING SAVED
                            imageCount += 1;
                            for(int m=0; m < huMoments.size() ; m++){
                                std::vector<float> featureVec(huMoments[m].begin(), huMoments[m].end());
                                char *csvName = (char *)"training.csv";
                                append_image_data_csv(csvName, arr, featureVec, 0);
                                for (int j = 0; j < 7; j++){
                                    cout<<huMoments[m][j]<<"     ";
                                }
                                cout<<endl;
                            }
                            break;
                        
            }
            if(exitFlag == 1){
                break;
            }
            imshow("TrainingData", res2);

            
        }
        delete capdev;
        destroyWindow("TrainingData"); // identifies a window
        return 0;
    });

    task6->Bind(wxEVT_BUTTON, [&](wxCommandEvent& event) {
        /*
            when the user presses button D the current frame is taken and the feature is created of the current object detected, then it is compared with the database and the  1 nearest is returned
            and that is the predicted category
            The object detected stays as it is till the next time the user clicks D
        */
        Mat Fram, res, colorFrame;
        VideoCapture *capdev;
        capdev = new VideoCapture(0);           //Opening Camera
        if( !capdev->isOpened() ) {
                printf("Unable to open video device\n");
                return(-1);
        }
        Size refS( (int) capdev->get(CAP_PROP_FRAME_WIDTH ),(int) capdev->get(CAP_PROP_FRAME_HEIGHT));
        namedWindow("Labelled", 1); // identifies a window
        namedWindow("Original", 1); // identifies a window
        char c;
        int imageCount = 1;
        while(1){
            *capdev >> colorFrame; // get a new frame from the camera, treat as a stream
            if( colorFrame.empty() ) {
                  printf("frame is empty\n");
                  break;
            }
            Mat cleanUp, res1, res2, res3, res4;
            thresholdBinary(colorFrame, cleanUp, 4);
            grow_shrink(cleanUp, res4, 2, 3, 1);
            grow_shrink(res4, res, 12, 3, 0);
            grow_shrink(res, res3, 10, 3, 1);
            vector<vector<double>> huMoments;
            conectedCompo(res3, res2, 0, huMoments);
            imshow("Labelled", res2);
            int exitFlag = 0;
            string LabelName = "";
            switch(c = waitKey(1)){
                case 'q':
                            exitFlag = 1;
                            break;
                case 'd':
                            char *csvName = (char *)"training.csv";
                            std::vector<std::vector<float>> featureVectors;
                            std::vector<char *> imageNames;
                            read_image_data_csv(csvName, imageNames, featureVectors, 0);
                            std::vector<float> featureVec(huMoments[0].begin(), huMoments[0].end());
                            float initialMinSum = sumOfSquares(featureVec, featureVectors[0]);
                            int indexLabel = 0;
                            for(int listOff = 1; listOff<featureVectors.size() ; listOff++){
                                float sumN = sumOfSquares(featureVec, featureVectors[listOff]);
                                if(sumN < initialMinSum){
                                    initialMinSum = sumN;
                                    indexLabel = listOff;
                                }
                            }
                            cout<<imageNames[indexLabel]<<endl;
                            LabelName = imageNames[indexLabel];
                            break;
            }
            putText(colorFrame, LabelName,Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 2, LINE_4);
            imshow("Original", colorFrame);
            if(exitFlag == 1){
                break;
            }
            
        }
        delete capdev;
        destroyWindow("Labelled");
        destroyWindow("Original");
        return 0;
    });
    Task7->Bind(wxEVT_BUTTON, [&](wxCommandEvent& event) {
        /*
            when the user presses button D the current frame is taken and the feature is created of the current object detected, then it is compared with the database and the 10 nearest is returned
            Similar to task 6 but the final prediction is done from max vote of the 10 nearest neighbours
        */
        Mat Fram, res, colorFrame;
        VideoCapture *capdev;
        capdev = new VideoCapture(0);           //Opening Camera
        if( !capdev->isOpened() ) {
                printf("Unable to open video device\n");
                return(-1);
        }
        Size refS( (int) capdev->get(CAP_PROP_FRAME_WIDTH ),(int) capdev->get(CAP_PROP_FRAME_HEIGHT));
        namedWindow("Labelled", 1); // identifies a window
        namedWindow("Original", 1); // identifies a window
        char c;
        int imageCount = 1;
        while(1){
            *capdev >> colorFrame; // get a new frame from the camera, treat as a stream
            if( colorFrame.empty() ) {
                  printf("frame is empty\n");
                  break;
            }
            Mat cleanUp, res1, res2, res3, res4;
            thresholdBinary(colorFrame, cleanUp, 4);
            grow_shrink(cleanUp, res4, 2, 3, 1);
            grow_shrink(res4, res, 12, 3, 0);
            grow_shrink(res, res3, 10, 3, 1);
            vector<vector<double>> huMoments;
            conectedCompo(res3, res2, 0, huMoments);
            imshow("Labelled", res2);
            int exitFlag = 0;
            switch(c = waitKey(1)){
                case 'q':
                            exitFlag = 1;
                            break;
                case 'd':
                            char *csvName = (char *)"training.csv";
                            std::vector<std::vector<float>> featureVectors;
                            std::vector<char *> imageNames;
                            read_image_data_csv(csvName, imageNames, featureVectors, 0);
                            std::vector<float> featureVec(huMoments[0].begin(), huMoments[0].end());
                            vector<string> topLabels;
                            vector<float> distanceMetric;
                            int count=0;

                            while(count < 5){ //taking default k as 10
                                topLabels.push_back(string(imageNames[count]));// to store label with the distance metric value
                                distanceMetric.push_back(sumOfSquares(featureVec, featureVectors[count]));
                                count++;
                            }
                            bubbleSortIndex(distanceMetric, topLabels);
                  
                            for(int k = 5; k<imageNames.size(); k++){
                                //check if the sum of squares is less than the maximum value
                                double sumOfSqua = sumOfSquares(featureVec, featureVectors[k]);
                                if(sumOfSqua < distanceMetric[5-1]){
                                    distanceMetric[5-1] = sumOfSqua;
                                    topLabels[5-1] = string(imageNames[k]);
                                    bubbleSortIndex(distanceMetric, topLabels);
                                }
                            }

                            string labelName = findLabelMost(topLabels);
                            for(int jk=0;jk<5;jk++){
                                cout<<topLabels[jk]<<"    ";
                            }
                            cout<<endl;
                            cout<<labelName;
                            putText(colorFrame, labelName,Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 2, LINE_4);
                            break;
            }
            imshow("Original", colorFrame);
            if(exitFlag == 1){
                break;
            }
            
        }
        delete capdev;
        destroyWindow("Labelled");
        destroyWindow("Original");
        return 0;
    });
    Task7_1->Bind(wxEVT_BUTTON, [&](wxCommandEvent& event) {
        /*
            If the distanceMetric dosent fall under a threshold then we say the object is not trained this is part of extension, the results are not that great though
        */
        Mat Fram, res;
        string fileName = string(LabelInput1->GetValue().mb_str());
        cout<<fileName<<endl;
        string image_path = samples::findFile(fileName,false);
        Mat colorFrame = imread(image_path, IMREAD_COLOR);
    
        namedWindow("Labelled", 1); // identifies a window
        namedWindow("Original", 1); // identifies a window
        char c;
        int imageCount = 1;
        while(1){
            //*capdev >> colorFrame; // get a new frame from the camera, treat as a stream
            if( colorFrame.empty() ) {
                  printf("frame is empty\n");
                  break;
            }
            Mat cleanUp, res1, res2, res3, res4;
            thresholdBinary(colorFrame, cleanUp, 4);
            grow_shrink(cleanUp, res4, 2, 3, 1);
            grow_shrink(res4, res, 12, 3, 0);
            grow_shrink(res, res3, 10, 3, 1);
            vector<vector<double>> huMoments;
            conectedCompo(res3, res2, 0, huMoments);
            imshow("Labelled", res2);
            int exitFlag = 0;
            string labelName = "";
            switch(c = waitKey(1)){
                case 'q':
                            exitFlag = 1;
                            break;
                case 'd':
                            char *csvName = (char *)"training.csv";
                            std::vector<std::vector<float>> featureVectors;
                            std::vector<char *> imageNames;
                            read_image_data_csv(csvName, imageNames, featureVectors, 0);
                            std::vector<float> featureVec(huMoments[0].begin(), huMoments[0].end());
                            vector<string> topLabels;
                            vector<float> distanceMetric;
                            int count=0;

                            while(count < 10){ //taking default k as 10
                                topLabels.push_back(string(imageNames[count]));// to store label with the distance metric value
                                distanceMetric.push_back(sumOfSquares(featureVec, featureVectors[count]));
                                count++;
                            }
                            bubbleSortIndex(distanceMetric, topLabels);
                  
                            for(int k = 10; k<imageNames.size(); k++){
                                //check if the sum of squares is less than the maximum value
                                double sumOfSqua = sumOfSquares(featureVec, featureVectors[k]);
                                if(sumOfSqua < distanceMetric[10-1]){
                                    distanceMetric[10-1] = sumOfSqua;
                                    topLabels[10-1] = string(imageNames[k]);
                                    bubbleSortIndex(distanceMetric, topLabels);
                                }
                            }
                            cout<<"Distance Metric  = "<<distanceMetric[0]<<endl;
                            if(distanceMetric[0] > 2){
                                labelName = "Not Trained";
                                break;
                            }
                            labelName = findLabelMost(topLabels);
                            for(int jk=0;jk<10;jk++){
                                cout<<topLabels[jk]<<"    ";
                            }
                            cout<<endl;
                            cout<<labelName;
                            break;
            }
            putText(colorFrame, labelName,Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 2, LINE_4);
            imshow("Original", colorFrame);
            if(exitFlag == 1){
                break;
            }
            
        }
        //delete capdev;
        destroyWindow("Labelled");
        destroyWindow("Original");
        return 0;
    });

    Task8->Bind(wxEVT_BUTTON, [&](wxCommandEvent& event) {
        /*
            Takes in a Test Data folder, the image name should contain the actual label of the object in the image and then prediction is done using the KNN K = 10 and confusion matrix is drawn
        */
        vector<string> labelLists = { "glasses", "airpodCase", "comb","ballPen","greenPen","batmanS","screw","heman","penDrive","clipLays","clipSmall","batmanBig","watch","multiPeeler","peelerSmall","scissor","holder","smallTrim","trimNose","hammer" };
        Mat Fram, res;

        string dirName = string(LabelInput2->GetValue().mb_str());
        dirName = pathToCompatibleString(dirName);

        cout<<dirName<<endl;

        char dirname[256];
        char filename[256];
        strcpy(dirname, dirName.c_str());      
        DIR *dirp;
        struct dirent *dp;

        dirp = opendir( dirname );
        if( dirp == NULL) {
            printf("Cannot open directory %s\n", dirname);
            exit(-1);
        }
        Mat colorFrame ;
        vector<string> predicted;
        vector<string> actual;
        vector<string> imageFileName;
        int countImages = 0;
        while( (dp = readdir(dirp)) != NULL ) {
            if( strstr(dp->d_name, ".jpg") ||   strstr(dp->d_name, ".png") ||   strstr(dp->d_name, ".ppm") ||   strstr(dp->d_name, ".tif") ) {
                // build the overall filename
                countImages++;
                strcpy(filename, dirname);
                strcat(filename, "//");
                strcat(filename, dp->d_name);
                imageFileName.push_back(string(dp->d_name));
                cout<<filename<<endl;
                colorFrame = imread(filename, IMREAD_COLOR);
                Mat cleanUp, res1, res2, res3, res4;
                thresholdBinary(colorFrame, cleanUp, 4);
                grow_shrink(cleanUp, res4, 2, 3, 1);
                grow_shrink(res4, res, 12, 3, 0);
                grow_shrink(res, res3, 10, 3, 1);
                vector<vector<double>> huMoments;
                conectedCompo(res3, res2, 0, huMoments);
                string labelName;
                char *csvName = (char *)"training.csv";
                std::vector<std::vector<float>> featureVectors;
                std::vector<char *> imageNames;
                read_image_data_csv(csvName, imageNames, featureVectors, 0);
                std::vector<float> featureVec(huMoments[0].begin(), huMoments[0].end());
                vector<string> topLabels;
                vector<float> distanceMetric;
                int count=0;

                while(count < 10){ //taking default k as 10
                    topLabels.push_back(string(imageNames[count]));// to store label with the distance metric value
                    distanceMetric.push_back(sumOfSquares(featureVec, featureVectors[count]));
                    count++;
                }
                bubbleSortIndex(distanceMetric, topLabels);
        
                for(int k = 10; k<imageNames.size(); k++){
                    //check if the sum of squares is less than the maximum value
                    //double sumOfSqua = sumOfSquares(featureVec, featureVectors[k]);
                    double sumOfSqua = sumOfSquares(featureVec, featureVectors[k]);

                    if(sumOfSqua < distanceMetric[10-1]){
                        distanceMetric[10-1] = sumOfSqua;
                        topLabels[10-1] = string(imageNames[k]);
                        bubbleSortIndex(distanceMetric, topLabels);
                    }
                }
                labelName = findLabelMost(topLabels);
                predicted.push_back(labelName);
                int matchCount=0; //for check
                string fName = string(dp->d_name);
                for(int labelI=0; labelI< labelLists.size();labelI++){
                    if (fName.find(labelLists[labelI]) != string::npos) {
                        actual.push_back(labelLists[labelI]);
                        matchCount++;
                    }
                }
                cout<<"Match Count = "<<matchCount<<endl;//for testing

            }
        }
        cout<<"Image Count = "<<countImages<<endl;
        int matchCount = 0;
        std::unordered_map<std::string, int> labelsIndexHas;
        for(int labelI=0; labelI< labelLists.size();labelI++){
            labelsIndexHas[labelLists[labelI]] = labelI;
        }
        vector<vector<int>> confusion(labelLists.size(), vector<int> (labelLists.size(), 0));
        for(int i=0;i<countImages;i++){
            if(predicted[i] == actual[i]){
                matchCount++;
            }
            confusion[labelsIndexHas[actual[i]]][labelsIndexHas[predicted[i]]]++;
        }
        float accuracy = (float(matchCount)/countImages)*100;
        cout<<"Accuracy = "<<(accuracy)<<endl;
        LabelInput3->SetValue(wxString::Format("Accuracy = %s", to_string(accuracy)));

        for(int i = 0;i<labelLists.size();i++){
            for(int j = 0;j<labelLists.size();j++){
                cout<<confusion[i][j]<<"    ";
            }
            cout<<endl;
        }
        std::ofstream out1("confusionMatrix.csv");

        for (auto& row : confusion) {
            for (auto col : row)
                out1 << col <<',';
            out1 << '\n';
        }


    });


}
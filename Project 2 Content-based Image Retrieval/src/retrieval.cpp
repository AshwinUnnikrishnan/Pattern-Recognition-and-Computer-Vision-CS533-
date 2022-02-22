#include <wx/filepicker.h>
#include <wx/sysopt.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include<fileLib.h>
#include <unistd.h>
#include <string>
#include <wx/wxprec.h>
#ifndef WX_PRECOMP
#include <wx/wx.h>
#endif

using namespace cv;
using namespace std;

string pathToCompatibleString(string pathV){
    std::replace(pathV.begin(), pathV.end(), '/', '|');
    size_t pos;
    while ((pos = pathV.find("|")) != std::string::npos) {
        pathV.replace(pos, 1, "//");
    }
    return pathV;
}
// Workaround : with wxWidgets version <= 3.1.4 wxFilePickerCtrl::SetFilterIndex doesn't work on macOS
class FilePickerCtrl : public wxFilePickerCtrl {

public:
  FilePickerCtrl(wxWindow *parent, wxWindowID id, const wxString& path = wxEmptyString, const wxString& message = wxFileSelectorPromptStr, const wxString& wildcard = wxFileSelectorDefaultWildcardStr, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxFLP_DEFAULT_STYLE, const wxValidator& validator = wxDefaultValidator, const wxString& name = wxFilePickerCtrlNameStr) : wxFilePickerCtrl(parent, id, path, message, wildcard, pos, size, style, validator, name) {
    //Used to pick the files like images, csv provided by the wxWidgets
    auto pickerCtrl = GetPickerCtrl();
    auto wx_dialog_style = 0;
    if ((style & wxFLP_OPEN) == wxFLP_OPEN) wx_dialog_style |= wxFD_OPEN;
    if ((style & wxFLP_SAVE) == wxFLP_SAVE) wx_dialog_style |= wxFD_SAVE;
    if ((style & wxFLP_OVERWRITE_PROMPT) == wxFLP_OVERWRITE_PROMPT) wx_dialog_style |= wxFD_OVERWRITE_PROMPT;
    if ((style & wxFLP_FILE_MUST_EXIST) == wxFLP_FILE_MUST_EXIST) wx_dialog_style |= wxFD_FILE_MUST_EXIST;
    if ((style & wxFLP_CHANGE_DIR) == wxFLP_CHANGE_DIR) wx_dialog_style |= wxFD_CHANGE_DIR;
    pickerCtrl->Bind(wxEVT_BUTTON, [=](wxCommandEvent& event) {
      wxFileDialog openFileDialog(parent, message, path, wxEmptyString, wildcard, wx_dialog_style);
      openFileDialog.SetFilterIndex(0);
      if (openFileDialog.ShowModal() == wxID_OK) {
        SetPath(openFileDialog.GetPath());
        wxPostEvent(this, wxFileDirPickerEvent(wxEVT_FILEPICKER_CHANGED, this, id, GetPath()));
      }
    });
  }
};

namespace Examples {
  class Frame : public wxFrame {
  public:
    Frame() : wxFrame(nullptr, wxID_ANY, "FilePickerCtrl example") {
      //below we are binding a handler to be called when imagePicker is clicked
    imagePicker->Bind(wxEVT_FILEPICKER_CHANGED, [&](wxFileDirPickerEvent& event) {
        label->SetLabel(wxString::Format("File = %s", imagePicker->GetPath()));
      });
      //below we are binding a handler to be called when csv is selected in the GUI
    featureCsvName->Bind(wxEVT_FILEPICKER_CHANGED, [&](wxFileDirPickerEvent& event) {
        featuresetLabel->SetLabel(wxString::Format("CSVFile = %s", featureCsvName->GetPath()));
      });
    //below we are binding a handler to be called when Photo folder selection is clicked
    imageFolder->Bind(wxEVT_DIRPICKER_CHANGED, [&](wxFileDirPickerEvent& event) {
        label1->SetLabel(wxString::Format("Path = %s", event.GetPath()));
      });
    // Below for each task a button and its handler is getting assigned
    // At first the handler will check if the image is selected and the photo folder is provided and it will only excute actual 
    //functionality ones these are given.

    //If CSV is not given it will check in the photos folder, if it is not present there as well, then the featureCreation function is called
    //In feature creation the features are generated and stored in the csv file provided
    //Once the feature creation function is called it calls the compare function based on the feature stored in the previous step and the top N 
    //Results are returned for the image


    taskOneButton->Bind(wxEVT_BUTTON, [&](wxCommandEvent& event) {
        string taskOneCSV, imagePath, task1Image;
        if(imagePicker->GetPath().compare("") && imageFolder->GetPath().compare("")){
            resultLabel->SetLabel(wxString::Format("Need to call my function and search for result here"));
            vector<string> result;
            if(featureCsvName->GetPath().compare("")== 0){
                //featureFolder Not Given
                //First check if present in photos directory if yes then read and continue
                //Else create and continue
                featuresetLabel->SetLabel(wxString::Format("Feature CSV not given checking in images folder and generating own csv if necessary"));
                taskOneCSV = pathToCompatibleString(imageFolder->GetPath().ToStdString() + "/taskOneFeatures.csv");
                char csvName[taskOneCSV.length()];
                strcpy(csvName, taskOneCSV.c_str());
                if (access(csvName, F_OK) == -1){
                    cout<<"File does not exist Calling the feature creation\n";
                    imagePath = pathToCompatibleString(imageFolder->GetPath().ToStdString() + "/");
                    readImagesExtractTaskOne(imagePath, taskOneCSV);
                }
                featuresetLabel->SetLabel((taskOneCSV));

            }
            else{
                taskOneCSV = pathToCompatibleString(featureCsvName->GetPath().ToStdString());
                //search for the similar and print output
            }
            task1Image = pathToCompatibleString(imagePicker->GetPath().ToStdString());

            //Read input from textlabel if empty string set to default 3
            int top = 3;
            string tempTop = topResults->GetValue().ToStdString();
            if(tempTop.compare("") != 0){
                top = stoi(tempTop);
            }
            result = readBackImageTaskOne(task1Image, taskOneCSV, top);
            string tempResult = "Matching Images are ";
            selectedImage->SetBitmap(wxImage(task1Image));
            selectedImage->SetSize(200,200);
            cout<<result.size()<<endl;
            int i = 0;
            for(i=0;i<result.size(); i++){
                //cout<<result[i]<<endl;
                tempResult += "\n" + result[i];
                if(i<7){
                    picList[i]->SetBitmap(wxImage(result[i]));
                    picList[i]->SetSize(150,150);
                }
            }
            while(i<3){
                picList[i]->SetBitmap(wxNullBitmap);
                i++;
            }
            resultLabel->SetLabel(tempResult);
        }
        else{
            if(imageFolder->GetPath().compare("") == 0){
                resultLabel->SetLabel(wxString::Format("Please select the image folder where you want to retrieve from "));
            }
            else{
                resultLabel->SetLabel(wxString::Format("Please select the image that you want to see similar one for "));
            }
        }
      });
    taskTwoButton->Bind(wxEVT_BUTTON, [&](wxCommandEvent& event) {
        string taskTwoCSV, imagePath, task1Image;
        if(imagePicker->GetPath().compare("") && imageFolder->GetPath().compare("")){
            resultLabel->SetLabel(wxString::Format("Need to call my function and search for result here"));
            vector<string> result;
            if(featureCsvName->GetPath().compare("")== 0){
                //featureFolder Not Given
                //First check if present in photos directory if yes then read and continue
                //Else create and continue
                featuresetLabel->SetLabel(wxString::Format("Feature CSV not given checking in images folder and generating own csv if necessary"));
                taskTwoCSV = pathToCompatibleString(imageFolder->GetPath().ToStdString() + "/taskTwoFeatures.csv");
                char csvName[taskTwoCSV.length()];
                strcpy(csvName, taskTwoCSV.c_str());
                if (access(csvName, F_OK) == -1){
                    cout<<"File does not exist Calling the feature creation\n";
                    imagePath = pathToCompatibleString(imageFolder->GetPath().ToStdString() + "/");
                    readImagesExtractTaskTwo(imagePath, taskTwoCSV);
                }
                featuresetLabel->SetLabel((taskTwoCSV));

            }
            else{
                taskTwoCSV = pathToCompatibleString(featureCsvName->GetPath().ToStdString());
                //search for the similar and print output
            }
            task1Image = pathToCompatibleString(imagePicker->GetPath().ToStdString());

            //Read input from textlabel if empty string set to default 3
            int top = 3;
            string tempTop = topResults->GetValue().ToStdString();
            if(tempTop.compare("") != 0){
                top = stoi(tempTop);
            }
            result = readBackImageTaskTwo(task1Image, taskTwoCSV, top);
            string tempResult = "Matching Images are ";
            selectedImage->SetBitmap(wxImage(task1Image));
            selectedImage->SetSize(200,200);
            cout<<result.size()<<endl;
            int i = 0;
            for(i=0;i<result.size(); i++){
                //cout<<result[i]<<endl;
                tempResult += "\n" + result[i];
                if(i<7){
                    picList[i]->SetBitmap(wxImage(result[i]));
                    picList[i]->SetSize(150,150);
                }
            }
            while(i<3){
                picList[i]->SetBitmap(wxNullBitmap);
                i++;
            }
            resultLabel->SetLabel(tempResult);
        }
        else{
            if(imageFolder->GetPath().compare("") == 0){
                resultLabel->SetLabel(wxString::Format("Please select the image folder where you want to retrieve from "));
            }
            else{
                resultLabel->SetLabel(wxString::Format("Please select the image that you want to see similar one for "));
            }
        }
      });  
    taskTwoExtensionButton->Bind(wxEVT_BUTTON, [&](wxCommandEvent& event) {
        string taskTwoCSV, imagePath, task1Image;
        if(imagePicker->GetPath().compare("") && imageFolder->GetPath().compare("")){
            resultLabel->SetLabel(wxString::Format("Need to call my function and search for result here"));
            vector<string> result;
            if(featureCsvName->GetPath().compare("")== 0){
                //featureFolder Not Given
                //First check if present in photos directory if yes then read and continue
                //Else create and continue
                featuresetLabel->SetLabel(wxString::Format("Feature CSV not given checking in images folder and generating own csv if necessary"));
                taskTwoCSV = pathToCompatibleString(imageFolder->GetPath().ToStdString() + "/taskTwoExtensionFeatures.csv");
                char csvName[taskTwoCSV.length()];
                strcpy(csvName, taskTwoCSV.c_str());
                if (access(csvName, F_OK) == -1){
                    cout<<"File does not exist Calling the feature creation\n";
                    imagePath = pathToCompatibleString(imageFolder->GetPath().ToStdString() + "/");
                    readImagesExtractTaskTwoExtension(imagePath, taskTwoCSV);
                }
                featuresetLabel->SetLabel((taskTwoCSV));

            }
            else{
                taskTwoCSV = pathToCompatibleString(featureCsvName->GetPath().ToStdString());
                //search for the similar and print output
            }
            task1Image = pathToCompatibleString(imagePicker->GetPath().ToStdString());

            //Read input from textlabel if empty string set to default 3
            int top = 3;
            string tempTop = topResults->GetValue().ToStdString();
            if(tempTop.compare("") != 0){
                top = stoi(tempTop);
            }
            result = readBackImageTaskTwoExtension(task1Image, taskTwoCSV, top);
            string tempResult = "Matching Images are ";
            selectedImage->SetBitmap(wxImage(task1Image));
            selectedImage->SetSize(200,200);
            cout<<result.size()<<endl;
            int i = 0;
            for(i=0;i<result.size(); i++){
                //cout<<result[i]<<endl;
                tempResult += "\n" + result[i];
                if(i<7){
                    picList[i]->SetBitmap(wxImage(result[i]));
                    picList[i]->SetSize(150,150);
                }
            }
            while(i<3){
                picList[i]->SetBitmap(wxNullBitmap);
                i++;
            }
            resultLabel->SetLabel(tempResult);
        }
        else{
            if(imageFolder->GetPath().compare("") == 0){
                resultLabel->SetLabel(wxString::Format("Please select the image folder where you want to retrieve from "));
            }
            else{
                resultLabel->SetLabel(wxString::Format("Please select the image that you want to see similar one for "));
            }
        }
      });  
    taskBlueBinExtensionButton->Bind(wxEVT_BUTTON, [&](wxCommandEvent& event) {
        string taskTwoCSV, imagePath, task1Image;
        if(imagePicker->GetPath().compare("") && imageFolder->GetPath().compare("")){
            resultLabel->SetLabel(wxString::Format("Need to call my function and search for result here"));
            vector<string> result;
            if(featureCsvName->GetPath().compare("")== 0){
                //featureFolder Not Given
                //First check if present in photos directory if yes then read and continue
                //Else create and continue
                featuresetLabel->SetLabel(wxString::Format("Feature CSV not given checking in images folder and generating own csv if necessary"));
                taskTwoCSV = pathToCompatibleString(imageFolder->GetPath().ToStdString() + "/taskBlueBin.csv");
                char csvName[taskTwoCSV.length()];
                strcpy(csvName, taskTwoCSV.c_str());
                if (access(csvName, F_OK) == -1){
                    cout<<"File does not exist Calling the feature creation\n";
                    imagePath = pathToCompatibleString(imageFolder->GetPath().ToStdString() + "/");
                    readImagesExtractTaskBlueBin(imagePath, taskTwoCSV);
                }
                featuresetLabel->SetLabel((taskTwoCSV));

            }
            else{
                taskTwoCSV = pathToCompatibleString(featureCsvName->GetPath().ToStdString());
                //search for the similar and print output
            }
            task1Image = pathToCompatibleString(imagePicker->GetPath().ToStdString());

            //Read input from textlabel if empty string set to default 3
            int top = 3;
            string tempTop = topResults->GetValue().ToStdString();
            if(tempTop.compare("") != 0){
                top = stoi(tempTop);
            }
            result = readBackImageTaskBlueBin(task1Image, taskTwoCSV, top);
            string tempResult = "Matching Images are ";
            selectedImage->SetBitmap(wxImage(task1Image));
            selectedImage->SetSize(200,200);
            cout<<result.size()<<endl;
            int i = 0;
            for(i=0;i<result.size(); i++){
                //cout<<result[i]<<endl;
                tempResult += "\n" + result[i];
                if(i<7){
                    picList[i]->SetBitmap(wxImage(result[i]));
                    picList[i]->SetSize(150,150);
                }
            }
            while(i<3){
                picList[i]->SetBitmap(wxNullBitmap);
                i++;
            }
            resultLabel->SetLabel(tempResult);
        }
        else{
            if(imageFolder->GetPath().compare("") == 0){
                resultLabel->SetLabel(wxString::Format("Please select the image folder where you want to retrieve from "));
            }
            else{
                resultLabel->SetLabel(wxString::Format("Please select the image that you want to see similar one for "));
            }
        }
      });  
    taskThreeButton->Bind(wxEVT_BUTTON, [&](wxCommandEvent& event) {
        string taskTwoCSV, imagePath, task1Image;
        if(imagePicker->GetPath().compare("") && imageFolder->GetPath().compare("")){
            resultLabel->SetLabel(wxString::Format("Need to call my function and search for result here"));
            vector<string> result;
            if(featureCsvName->GetPath().compare("")== 0){
                //featureFolder Not Given
                //First check if present in photos directory if yes then read and continue
                //Else create and continue
                featuresetLabel->SetLabel(wxString::Format("Feature CSV not given checking in images folder and generating own csv if necessary"));
                taskTwoCSV = pathToCompatibleString(imageFolder->GetPath().ToStdString() + "/taskThreeFeatures.csv");
                char csvName[taskTwoCSV.length()];
                strcpy(csvName, taskTwoCSV.c_str());
                if (access(csvName, F_OK) == -1){
                    cout<<"File does not exist Calling the feature creation\n";
                    imagePath = pathToCompatibleString(imageFolder->GetPath().ToStdString() + "/");
                    readImagesExtractTaskThree(imagePath, taskTwoCSV);
                }
                featuresetLabel->SetLabel((taskTwoCSV));

            }
            else{
                taskTwoCSV = pathToCompatibleString(featureCsvName->GetPath().ToStdString());
                //search for the similar and print output
            }
            task1Image = pathToCompatibleString(imagePicker->GetPath().ToStdString());

            //Read input from textlabel if empty string set to default 3
            int top = 3;
            string tempTop = topResults->GetValue().ToStdString();
            if(tempTop.compare("") != 0){
                top = stoi(tempTop);
            }
            result = readBackImageTaskThree(task1Image, taskTwoCSV, top);
            string tempResult = "Matching Images are ";
            selectedImage->SetBitmap(wxImage(task1Image));
            selectedImage->SetSize(200,200);
            cout<<result.size()<<endl;
            int i = 0;
            for(i=0;i<result.size(); i++){
                //cout<<result[i]<<endl;
                tempResult += "\n" + result[i];
                if(i<7){
                    picList[i]->SetBitmap(wxImage(result[i]));
                    picList[i]->SetSize(150,150);
                }
            }
            while(i<3){
                picList[i]->SetBitmap(wxNullBitmap);
                i++;
            }
            resultLabel->SetLabel(tempResult);
        }
        else{
            if(imageFolder->GetPath().compare("") == 0){
                resultLabel->SetLabel(wxString::Format("Please select the image folder where you want to retrieve from "));
            }
            else{
                resultLabel->SetLabel(wxString::Format("Please select the image that you want to see similar one for "));
            }
        }
      });  
    taskFourButton->Bind(wxEVT_BUTTON, [&](wxCommandEvent& event) {
        string taskTwoCSV, imagePath, task1Image;
        if(imagePicker->GetPath().compare("") && imageFolder->GetPath().compare("")){
            resultLabel->SetLabel(wxString::Format("Need to call my function and search for result here"));
            vector<string> result;
            if(featureCsvName->GetPath().compare("")== 0){
                //featureFolder Not Given
                //First check if present in photos directory if yes then read and continue
                //Else create and continue
                featuresetLabel->SetLabel(wxString::Format("Feature CSV not given checking in images folder and generating own csv if necessary"));
                taskTwoCSV = pathToCompatibleString(imageFolder->GetPath().ToStdString() + "/taskFourFeatures.csv");
                char csvName[taskTwoCSV.length()];
                strcpy(csvName, taskTwoCSV.c_str());
                if (access(csvName, F_OK) == -1){
                    cout<<"File does not exist Calling the feature creation\n";
                    imagePath = pathToCompatibleString(imageFolder->GetPath().ToStdString() + "/");
                    readImagesExtractTaskFour(imagePath, taskTwoCSV);
                }
                featuresetLabel->SetLabel((taskTwoCSV));

            }
            else{
                taskTwoCSV = pathToCompatibleString(featureCsvName->GetPath().ToStdString());
                //search for the similar and print output
            }
            task1Image = pathToCompatibleString(imagePicker->GetPath().ToStdString());

            //Read input from textlabel if empty string set to default 3
            int top = 3;
            string tempTop = topResults->GetValue().ToStdString();
            if(tempTop.compare("") != 0){
                top = stoi(tempTop);
            }
            result = readBackImageTaskFour(task1Image, taskTwoCSV, top);
            string tempResult = "Matching Images are ";
            selectedImage->SetBitmap(wxImage(task1Image));
            selectedImage->SetSize(200,200);
            cout<<result.size()<<endl;
            int i = 0;
            for(i=0;i<result.size(); i++){
                //cout<<result[i]<<endl;
                tempResult += "\n" + result[i];
                if(i<7){
                    picList[i]->SetBitmap(wxImage(result[i]));
                    picList[i]->SetSize(150,150);
                }
            }
            while(i<7){
                picList[i]->SetBitmap(wxNullBitmap);
                i++;
            }
            resultLabel->SetLabel(tempResult);
        }
        else{
            if(imageFolder->GetPath().compare("") == 0){
                resultLabel->SetLabel(wxString::Format("Please select the image folder where you want to retrieve from "));
            }
            else{
                resultLabel->SetLabel(wxString::Format("Please select the image that you want to see similar one for "));
            }
        }
      });  
    taskFiveButton->Bind(wxEVT_BUTTON, [&](wxCommandEvent& event) {
        string taskTwoCSV, imagePath, task1Image;
        if(imagePicker->GetPath().compare("") && imageFolder->GetPath().compare("")){
            resultLabel->SetLabel(wxString::Format("Need to call my function and search for result here"));
            vector<string> result;
            if(featureCsvName->GetPath().compare("")== 0){
                //featureFolder Not Given
                //First check if present in photos directory if yes then read and continue
                //Else create and continue
                featuresetLabel->SetLabel(wxString::Format("Feature CSV not given checking in images folder and generating own csv if necessary"));
                taskTwoCSV = pathToCompatibleString(imageFolder->GetPath().ToStdString() + "/taskFiveFeatures.csv");
                char csvName[taskTwoCSV.length()];
                strcpy(csvName, taskTwoCSV.c_str());
                if (access(csvName, F_OK) == -1){
                    cout<<"File does not exist Calling the feature creation\n";
                    imagePath = pathToCompatibleString(imageFolder->GetPath().ToStdString() + "/");
                    readImagesExtractTaskFive(imagePath, taskTwoCSV);
                }
                featuresetLabel->SetLabel((taskTwoCSV));

            }
            else{
                taskTwoCSV = pathToCompatibleString(featureCsvName->GetPath().ToStdString());
                //search for the similar and print output
            }
            task1Image = pathToCompatibleString(imagePicker->GetPath().ToStdString());

            //Read input from textlabel if empty string set to default 3
            int top = 3;
            string tempTop = topResults->GetValue().ToStdString();
            if(tempTop.compare("") != 0){
                top = stoi(tempTop);
            }
            result = readBackImageTaskFive(task1Image, taskTwoCSV, top);
            string tempResult = "Matching Images are ";
            selectedImage->SetBitmap(wxImage(task1Image));
            selectedImage->SetSize(200,200);
            cout<<result.size()<<endl;
            int i = 0;
            for(i=0;i<result.size(); i++){
                //cout<<result[i]<<endl;
                tempResult += "\n" + result[i];
                if(i<7){
                    picList[i]->SetBitmap(wxImage(result[i]));
                    picList[i]->SetSize(150,150);
                }
            }
            while(i<7){
                picList[i]->SetBitmap(wxNullBitmap);
                i++;
            }
            resultLabel->SetLabel(tempResult);
        }
        else{
            if(imageFolder->GetPath().compare("") == 0){
                resultLabel->SetLabel(wxString::Format("Please select the image folder where you want to retrieve from "));
            }
            else{
                resultLabel->SetLabel(wxString::Format("Please select the image that you want to see similar one for "));
            }
        }
      });
    taskFourExtensionButton->Bind(wxEVT_BUTTON, [&](wxCommandEvent& event) {
        string taskTwoCSV, imagePath, task1Image;
        if(imagePicker->GetPath().compare("") && imageFolder->GetPath().compare("")){
            resultLabel->SetLabel(wxString::Format("Need to call my function and search for result here"));
            vector<string> result;
            if(featureCsvName->GetPath().compare("")== 0){
                //featureFolder Not Given
                //First check if present in photos directory if yes then read and continue
                //Else create and continue
                featuresetLabel->SetLabel(wxString::Format("Feature CSV not given checking in images folder and generating own csv if necessary"));
                taskTwoCSV = pathToCompatibleString(imageFolder->GetPath().ToStdString() + "/taskFourExtensionFeatures.csv");
                char csvName[taskTwoCSV.length()];
                strcpy(csvName, taskTwoCSV.c_str());
                if (access(csvName, F_OK) == -1){
                    cout<<"File does not exist Calling the feature creation\n";
                    imagePath = pathToCompatibleString(imageFolder->GetPath().ToStdString() + "/");
                    readImagesExtractTaskFourExtension(imagePath, taskTwoCSV);
                }
                featuresetLabel->SetLabel((taskTwoCSV));

            }
            else{
                taskTwoCSV = pathToCompatibleString(featureCsvName->GetPath().ToStdString());
                //search for the similar and print output
            }
            task1Image = pathToCompatibleString(imagePicker->GetPath().ToStdString());

            //Read input from textlabel if empty string set to default 3
            int top = 3;
            string tempTop = topResults->GetValue().ToStdString();
            if(tempTop.compare("") != 0){
                top = stoi(tempTop);
            }
            result = readBackImageTaskFourExtension(task1Image, taskTwoCSV, top);
            string tempResult = "Matching Images are ";
            selectedImage->SetBitmap(wxImage(task1Image));
            selectedImage->SetSize(200,200);
            int i = 0;
            for(i=0;i<result.size(); i++){
                //cout<<result[i]<<endl;
                tempResult += "\n" + result[i];
                if(i<7){
                    picList[i]->SetBitmap(wxImage(result[i]));
                    picList[i]->SetSize(150,150);
                }
            }
            while(i<7){
                picList[i]->SetBitmap(wxNullBitmap);
                i++;
            }
            resultLabel->SetLabel(tempResult);
        }
        else{
            if(imageFolder->GetPath().compare("") == 0){
                resultLabel->SetLabel(wxString::Format("Please select the image folder where you want to retrieve from "));
            }
            else{
                resultLabel->SetLabel(wxString::Format("Please select the image that you want to see similar one for "));
            }
        }
      });
    }
    
  private:
    //Below The items buttons, directory selection etc, GUI related declarations are being done
    wxPanel* panel = new wxPanel(this);
    //Getting number of images to search for top N 
    wxStaticText* topLabel = new wxStaticText(panel, wxID_ANY, "Select Number Of Results To Return", wxPoint(600, 10));
    wxTextCtrl* topResults = new wxTextCtrl(panel, wxID_ANY, "", {600, 40});



    wxStaticText* label = new wxStaticText(panel, wxID_ANY, "Image File To Search = ", wxPoint(10, 70));
    wxStaticText* selectImageLabel = new wxStaticText(panel, wxID_ANY, "Image Search Select", wxPoint(10, 10));

    //wxFilePickerCtrl* picker = new wxFilePickerCtrl(panel, wxID_ANY, wxEmptyString, wxEmptyString, "Text Files (*.txt)|*.txt|All Files (*.*)|*.*", {10, 10}, wxDefaultSize, wxFLP_DEFAULT_STYLE|wxFLP_SMALL);
    FilePickerCtrl* imagePicker = new FilePickerCtrl(panel, wxID_ANY, wxEmptyString, wxEmptyString, "Text Files (*.jpg)|*.jpg|All Files (*.*)|*.*", {10, 40}, wxDefaultSize, wxFLP_DEFAULT_STYLE|wxFLP_SMALL);
    //wxPanel* panel1 = new wxPanel(this);
    wxStaticText* label1 = new wxStaticText(panel, wxID_ANY, "Folder where the images are = ", wxPoint(10, 100));
    wxStaticText* selectFolderLabel = new wxStaticText(panel, wxID_ANY, "Select Photos Folder", wxPoint(200, 10));

    wxDirPickerCtrl* imageFolder = new wxDirPickerCtrl(panel, wxID_ANY, wxEmptyString, wxEmptyString, {200, 40}, wxDefaultSize, wxDIRP_DEFAULT_STYLE|wxDIRP_SMALL);
    
    wxStaticText* featuresetLabel = new wxStaticText(panel, wxID_ANY, "Feature CSV File To Use = ", wxPoint(10, 130));
    wxStaticText* selectCSVLabel = new wxStaticText(panel, wxID_ANY, "Select CSV", wxPoint(400, 10));

    //Check if csv given if not search current directory based on the task and if present use it else create csv and use it
    FilePickerCtrl* featureCsvName = new FilePickerCtrl(panel, wxID_ANY, wxEmptyString, wxEmptyString, "Text Files (*.csv)|*.csv|All Files (*.*)|*.*", {400, 40}, wxDefaultSize, wxFLP_DEFAULT_STYLE|wxFLP_SMALL);

    //Creating Task1 Button
    int resultCount = 3;//Need to take this from user

    wxButton* taskOneButton = new wxButton(panel, wxID_ANY, "Task1", {10, 160});
    wxButton* taskTwoButton = new wxButton(panel, wxID_ANY, "Task2", {100, 160});
    wxButton* taskThreeButton = new wxButton(panel, wxID_ANY, "Task3", {200, 160});
    wxButton* taskFourButton = new wxButton(panel, wxID_ANY, "Task4", {300, 160});
    wxButton* taskTwoExtensionButton = new wxButton(panel, wxID_ANY, "Task2Extension", {400, 160});
    wxButton* taskBlueBinExtensionButton = new wxButton(panel, wxID_ANY, "Task5", {550, 160});
    wxButton* taskFiveButton = new wxButton(panel, wxID_ANY, "Task5_2", {650, 160});
    wxButton* taskFourExtensionButton = new wxButton(panel, wxID_ANY, "Task4Extension", {850, 160});
    vector<wxStaticBitmap*> picList{new wxStaticBitmap(panel, wxID_ANY, wxNullBitmap, {10, 700}, wxDefaultSize),new wxStaticBitmap(panel, wxID_ANY, wxNullBitmap, {200, 700}, wxDefaultSize), new wxStaticBitmap(panel, wxID_ANY, wxNullBitmap, {360, 700}, wxDefaultSize), new wxStaticBitmap(panel, wxID_ANY, wxNullBitmap, {520, 700}, wxDefaultSize), new wxStaticBitmap(panel, wxID_ANY, wxNullBitmap, {680, 700}, wxDefaultSize), new wxStaticBitmap(panel, wxID_ANY, wxNullBitmap, {850, 700}, wxDefaultSize), new wxStaticBitmap(panel, wxID_ANY, wxNullBitmap, {1060, 700}, wxDefaultSize)};
    wxStaticBitmap* selectedImage = new wxStaticBitmap(panel, wxID_ANY, wxNullBitmap, {10, 500}, wxDefaultSize);
    wxStaticText* resultLabel = new wxStaticText(panel, wxID_ANY, "Result = ", wxPoint(10, 230));

 
  };

  class Application : public wxApp {
    bool OnInit() override {
      wxInitAllImageHandlers();
      wxSystemOptions::SetOption("osx.openfiledialog.always-show-types", 1);
      Frame* page = (new Frame());
      page->Show();
      return true;
    }
  };
}

wxIMPLEMENT_APP(Examples::Application);
// Main.cpp

#include <opencv2/opencv.hpp>

#include <iostream>
#include<conio.h>           // may have to modify this line if not using Windows

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main() {
    
    // open the original image, show an error message and bail if not successful
    cv::Mat imgOriginal = cv::imread("cards.png");
    if (imgOriginal.empty()) {
        std::cout << "\n" << "error reading image from file" << "\n\n";
        _getch();             // may have to modify this line if not using Windows
        return(0);
    }
    cv::imshow("1 - imgOriginal", imgOriginal);

    // change the background from white to black, since that will help later to extract better results during the use of Distance Transform
    // for every pixel . . .
    for (int x = 0; x < imgOriginal.rows; x++) {
        for (int y = 0; y < imgOriginal.cols; y++) {
            // if the current pixel is white, change it to black
            if (imgOriginal.at<cv::Vec3b>(x, y) == cv::Vec3b(255, 255, 255)) {
                imgOriginal.at<cv::Vec3b>(x, y)[0] = 0;
                imgOriginal.at<cv::Vec3b>(x, y)[1] = 0;
                imgOriginal.at<cv::Vec3b>(x, y)[2] = 0;
            }
        }
    }    
    cv::imshow("2 - imgOriginal with black background", imgOriginal);

    // sharpen the image

    
    // Create a kernel that we will use for accuting/sharpening our image
    // An approximation of second derivative, a quite strong kernel, do the laplacian filtering as it is
    // well, we need to convert everything in something more deeper then CV_8U because the kernel has some negative values,
    // and we can expect in general to have a Laplacian image with negative values BUT a 8bits unsigned int (the one we are working with)
    // can contain values from 0 to 255 so the possible negative number will be truncated
    cv::Mat matKernel = (cv::Mat_<float>(3, 3) << 1, 1, 1,
                                                  1, -8, 1,
                                                  1, 1, 1);

    // ToDo: is imgLaplacian ever used ??
    cv::Mat imgLaplacian;
    cv::Mat imgSharp = imgOriginal; // copy source image to another temporary one
    cv::filter2D(imgSharp, imgLaplacian, CV_32F, matKernel);
    imgOriginal.convertTo(imgSharp, CV_32F);
    cv::Mat imgSharpened = imgSharp - imgLaplacian;
    // convert back to 8bits gray scale
    imgSharpened.convertTo(imgSharpened, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
    // imshow( "Laplace Filtered Image", imgLaplacian );
    imshow("3 - New Sharped Image", imgSharpened);
    imgOriginal = imgSharpened; // copy back
    
    // convert to grayscale
    cv::Mat imgGrayscale;
    cvtColor(imgOriginal, imgGrayscale, CV_BGR2GRAY);
    cv::imshow("4 - imgGrayscale", imgGrayscale);

    // threshold
    cv::Mat imgThresh;
    cv::threshold(imgGrayscale, imgThresh, 40.0, 255.0, CV_THRESH_BINARY | CV_THRESH_OTSU);
    cv::imshow("5 - imgThresh", imgThresh);

    // Perform the distance transform algorithm
    cv::Mat imgDistTransResult;
    cv::distanceTransform(imgThresh, imgDistTransResult, CV_DIST_L2, 3);
    // normalize the distance image for range = {0.0, 1.0} so we can visualize and threshold it
    cv::normalize(imgDistTransResult, imgDistTransResult, 0, 1., cv::NORM_MINMAX);
    cv::imshow("6 - imgDistTransResult", imgDistTransResult);

    // threshold to obtain the peaks, these will be the markers for the foreground objects
    cv::threshold(imgDistTransResult, imgDistTransResult, 0.4, 1.0, CV_THRESH_BINARY);

    // dilate to find peaks
    cv::Mat kernel1 = cv::Mat::ones(3, 3, CV_8UC1);
    cv::dilate(imgDistTransResult, imgDistTransResult, kernel1);
    cv::imshow("7a - imgDistTransResult dilated", imgDistTransResult);

    // create the CV_8U version of the distance image, needed for findContours()
    cv::Mat imgDistTransResult8U;
    imgDistTransResult.convertTo(imgDistTransResult8U, CV_8U);
    cv::imshow("7b - imgDistTransResult dilated", imgDistTransResult);

    // Find total markers
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(imgDistTransResult8U, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    std::cout << "\n" << "contours.size() = " << contours.size() << "\n\n";

    // create the marker image, this will be a pass by reference output to watershed(), this has to be the same size as the others, but type CV_32SC1
    cv::Mat imgMarkers = cv::Mat::zeros(imgDistTransResult.size(), CV_32SC1);
    
    // draw the foreground markers in shades of gray on the markers image
    for (size_t i = 0; i < contours.size(); i++) {
        cv::drawContours(imgMarkers, contours, static_cast<int>(i), cv::Scalar::all(static_cast<int>(i) + 1), -1);
    }    

    // draw the background marker (small circle in the very top left) on the markers image
    cv::circle(imgMarkers, cv::Point(5, 5), 3, cv::Scalar(255.0, 255.0, 255.0), -1);
    cv::imshow("8 - imgMarkers just before watershed()", imgMarkers * 10000);

    // Perform the watershed algorithm
    cv::watershed(imgOriginal, imgMarkers);

    // declare an 8-bit 1-channel version of the markers image, then convert the 
    cv::Mat imgMarkers8U = cv::Mat::zeros(imgMarkers.size(), CV_8UC1);
    imgMarkers.convertTo(imgMarkers8U, CV_8UC1);
    // invert image
    cv::bitwise_not(imgMarkers8U, imgMarkers8U);
    cv::imshow("9 - imgMarkers8U", imgMarkers8U);
    
    // Generate random colors
    std::vector<cv::Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++) {
        int b = cv::theRNG().uniform(0, 255);
        int g = cv::theRNG().uniform(0, 255);
        int r = cv::theRNG().uniform(0, 255);
        colors.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
    // Create the result image
    cv::Mat imgFinalResult = cv::Mat::zeros(imgMarkers.size(), CV_8UC3);
    // Fill labeled objects with random colors
    for (int i = 0; i < imgMarkers.rows; i++) {
        for (int j = 0; j < imgMarkers.cols; j++) {
            int index = imgMarkers.at<int>(i, j);
            if (index > 0 && index <= static_cast<int>(contours.size())) {
                imgFinalResult.at<cv::Vec3b>(i, j) = colors[index - 1];
            } else {
                imgFinalResult.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
            }
        }
    }

    // Visualize the final image
    cv::imshow("10 - Final Result", imgFinalResult);
    cv::waitKey();
    return 0;
}


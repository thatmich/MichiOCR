#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdint.h>

using namespace cv;
using namespace std;

int const max_value = 255;
int const max_BINARY_value = 255;
float const bK = 0.5;
float const bR = 128;

const double PI = 3.141592653589793238460;

//Mat image = imread("C:\\Users\\wwwsu\\Downloads\\IMG_20200226_084626.jpg");
Mat image = imread("C:\\Users\\wwwsu\\Desktop\\AdvRoboOCR\\ConsoleApplication2\\x64\\Debug\\IMG_20200210_102347.jpg", IMREAD_GRAYSCALE);;
Mat dst;
/// Function Headers
Mat binarization(Mat img);
float avgColor(Mat img);
float sDeviation(Mat img);
void straightening(Mat s_img);

int main()
{
	namedWindow("mywin", WINDOW_NORMAL);
	resizeWindow("mywin", 274*1.8, 365*1.8);
	///resizeWindow("mywin", 365 * 2, 274*2);
	image = binarization(image);
	straightening(image);
	imshow("mywin", dst);


	waitKey(0);
}

Mat binarization(Mat img) {
	float threshVal;

	threshVal = avgColor(img) * (1 + bK * (sDeviation(img)/bR - 1));

	threshold(img, dst, threshVal, max_BINARY_value, 1);

	return dst;
	
}

float avgColor(Mat img) {
	int avgcount = 0;
	float avgtotal = 0;
	float avg = 0;
	for (int r = 0; r < img.rows; r++) {
		for (int c = 0; c < img.cols; c++) {
			avgtotal += img.at<uint8_t>(r, c);
			avgcount++;
		}
	}
	///printf("%d", avgcount);
	avg = avgtotal / avgcount;
	return avg;
}

float sDeviation(Mat img) {
	float mean, standardDeviation = 0.0;
	mean = avgColor(img);
	int r = 0, c = 0;
	for (r = 0; r < img.rows; r++) {
		for (c = 0; c < img.cols; c++) {
			standardDeviation += pow(img.at<uint8_t>(r, c) - mean, 2);
		}
	}
	///printf("%d", c*r);
	return sqrt(standardDeviation / (c*r));

}

void straightening(Mat s_img) {
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat canny_output, bounding_mat;
	int thresh = 100;
	Canny(s_img, canny_output, thresh, thresh * 2, 3);
	findContours(canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	std::vector<cv::Point> points;

	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);

	vector<RotatedRect> minRect(contours.size());

	for (int i = 0; i < contours.size(); i += 1)
	{
		minRect[i] = minAreaRect(Mat(contours[i]));
	}

	vector<Point2f> allpts;
	
	for (int i = 0; i < minRect.size(); i++)
	{
		Point2f p1[4];
		minRect[i].points(p1);
		allpts.push_back(p1[0]);
		allpts.push_back(p1[1]);
		allpts.push_back(p1[2]);
		allpts.push_back(p1[3]);
	}
	RotatedRect final = minAreaRect(allpts);
	Point2f vertices[4];
	final.points(vertices);


	//draw

	for (int i = 0; i < 4; i++)
		line(drawing, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0));

	namedWindow("win", WINDOW_NORMAL);
	resizeWindow("win", 274 * 1.8, 365 * 1.8);
	imshow("win", drawing);
}


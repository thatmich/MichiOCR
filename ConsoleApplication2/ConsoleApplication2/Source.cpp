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
float const bR = 100;

const double PI = 3.141592653589793238460;

//Mat image = imread("C:\\Users\\wwwsu\\Source\\Repos\\MichiOCR\\ConsoleApplication2\\IMG_20200311_105406.jpg", IMREAD_GRAYSCALE);
Mat image = imread("C:\\Users\\wwwsu\\Source\\Repos\\MichiOCR\\ConsoleApplication2\\IMG_20200210_102347.jpg", IMREAD_GRAYSCALE);
Mat dst = image;
/// Function Headers
Mat binarization(Mat img);
float avgColor(Mat img);
float sDeviation(Mat img);
Mat straightening(Mat s_img);
Mat rotate(Mat src, float angle);
Mat skeleton(Mat src);

float rotated_angle;

int main()
{
	
	
	dst = binarization(image);
	namedWindow("Binarized", WINDOW_NORMAL);
	//resizeWindow("mywin", 365 * 1.8, 274 * 1.8);
	resizeWindow("Binarized", 493, 657);
	imshow("Binarized", image);

	dst = straightening(dst);
	namedWindow("Pre-Straightened", WINDOW_NORMAL);
	//resizeWindow("mywin", 365 * 1.8, 274 * 1.8);
	resizeWindow("Pre-Straightened", 493, 657);
	imshow("Pre-Straightened", dst);

	Mat thing = rotate(image, rotated_angle);
	namedWindow("Post-Straightened", WINDOW_NORMAL);
	//resizeWindow("mywin", 365 * 1.8, 274 * 1.8);
	resizeWindow("Post-Straightened", 493, 657);
	imshow("Post-Straightened", thing);


	waitKey(0);
}

Mat binarization(Mat img) {
	float threshVal;
	threshVal = avgColor(img) * (1 + bK * (sDeviation(img) / bR - 1));

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

Mat straightening(Mat s_img) {
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
	RotatedRect rRect = RotatedRect(vertices[0], vertices[1], vertices[2]);
	rRect.points(vertices);


	float blob_angle_deg = rRect.angle;
	if (rRect.size.width < rRect.size.height) {
		printf("True \n");
		blob_angle_deg = 90 + blob_angle_deg;
	}


	//draw

	Mat rgb;
	cvtColor(image, rgb, COLOR_GRAY2BGR); //adds color
	
	for (int i = 0; i < 4; i++) {
		printf("Point %d: %f, %f\n", i, vertices[i].x, vertices[i].y);
		line(rgb, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0),5);
	}
	printf("Angle %f", blob_angle_deg);
	rotated_angle = blob_angle_deg;
	return rgb;
}

Mat rotate(Mat src, float angle) {
	Mat new_rot_dst;
	angle = -(90 - angle);
	// code referenced from Lars Schillingmann on 
	//   https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c
	// get rotation matrix for rotating the image around its center in pixel coordinates
	cv::Point2f center((src.cols - 1) / 2.0, (src.rows - 1) / 2.0);
	cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
	// determine bounding rectangle, center not relevant
	cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), src.size(), angle).boundingRect2f();
	// adjust transformation matrix
	rot.at<double>(0, 2) += bbox.width / 2.0 - src.cols / 2.0;
	rot.at<double>(1, 2) += bbox.height / 2.0 - src.rows / 2.0;

	cv::warpAffine(src, new_rot_dst, rot, bbox.size());

	return new_rot_dst;

}

Mat skeleton(Mat src) {

}
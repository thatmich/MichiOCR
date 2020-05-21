#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdint.h>

using namespace cv;
using namespace std;

int const max_value = 255;
int const max_BINARY_value = 255;
float const bK = 0.6;
float const bR = 90;

const double PI = 3.141592653589793238460;

Mat image = imread("C:\\Users\\wwwsu\\Source\\Repos\\MichiOCR\\ConsoleApplication2\\timesA.png", IMREAD_GRAYSCALE);
//Mat image = imread("C:\\Users\\wwwsu\\Source\\Repos\\MichiOCR\\ConsoleApplication2\\IMG_20200210_102347.jpg", IMREAD_GRAYSCALE);
Mat dst = image;
/// Function Headers
Mat binarization(Mat img);
float avgColor(Mat img);
float sDeviation(Mat img);
Mat straightening(Mat s_img);
Mat rotate(Mat src, float angle);
void thinning(cv::Mat& i);

void pruning(cv::Mat& im);
Mat dilate(Mat src);

int currentX = 0;
int currentY = 0;

Mat thing;
// Trackbar call back function 
static void onChangeZ(int pos, void* userInput);
static void onChangeX(int pos, void* userInput);
static void onChangeY(int pos, void* userInput);

float rotated_angle;

int main()
{
	
	
	dst = binarization(image);
	//namedWindow("Binarized", WINDOW_AUTOSIZE);
	////resizeWindow("mywin", 365 * 1.8, 274 * 1.8);
	//resizeWindow("Binarized", 493, 657);
	//imshow("Binarized", image);

	dst = straightening(dst);
	//namedWindow("Pre-Straightened", WINDOW_NORMAL);
	////resizeWindow("mywin", 365 * 1.8, 274 * 1.8);
	//resizeWindow("Pre-Straightened", 493, 657);
	//imshow("Pre-Straightened", dst);

	thing = rotate(image, rotated_angle);
	//namedWindow("Post-Straightened", WINDOW_NORMAL);
	////resizeWindow("mywin", 365 * 1.8, 274 * 1.8);
	//resizeWindow("Post-Straightened", 493, 657);
	//imshow("Post-Straightened", thing);

	thinning(thing);
	namedWindow("Pre Prune", WINDOW_NORMAL);
	resizeWindow("Pre Prune", thing.cols*3 +200, thing.rows*3 +200);
	imshow("Pre Prune", thing);
	pruning(thing);
	namedWindow("Post Prune", WINDOW_NORMAL);
	resizeWindow("Post Prune", thing.cols * 3 + 200, thing.rows * 3 + 200);
	imshow("Post Prune", thing);


	//namedWindow("Control Panel", WINDOW_NORMAL);
	//resizeWindow("Control", 300, 600);
	//int scale1 = 100;
	//int scaleX = thing.cols*3;
	//int scaleY = thing.rows*3;
	//// create a trackbar 
	//createTrackbar("Zoom", "Control Panel", &scale1, 1000, onChangeZ);
	//onChangeZ(0, 0);
	//createTrackbar("X", "Control Panel", &scaleX, 1000, onChangeX);
	//onChangeX(0, 0);
	//createTrackbar("Y", "Control Panel", &scaleY, 1000, onChangeY);
	//onChangeX(0, 0);
	//moveWindow("Control Panel", 0, 650);
	//imshow("Control Panel", 0);


	//thing = dilate(thing);
	//namedWindow("Dilate", WINDOW_AUTOSIZE);
	////resizeWindow("mywin", 365 * 1.8, 274 * 1.8);
	//resizeWindow("Dilate", 493, 657);
	//imshow("Dilate", true);

	////thing = prune(thing);
	//namedWindow("Prune", WINDOW_AUTOSIZE);
	////resizeWindow("mywin", 365 * 1.8, 274 * 1.8);
	//resizeWindow("Dilate", 493, 657);
	//imshow("Dilate", thing);
	waitKey(0);
}


// Trackbar call back function 
static void onChangeZ(int slider, void* userInput) {
	float scale = float(slider + 1) / 100;

	Mat img_converted;
	Size s(thing.size().width*scale, (int)(thing.size().height)*scale);
	resize(thing, img_converted, s);
	imshow("Scaled", img_converted);
}

static void onChangeX(int slider, void* userInput) {
	currentX = -slider;
	Mat trans_mat = (Mat_<double>(2, 3) << 1, 0, currentX, 0, 1, currentY);
	Mat img_converted;
	warpAffine(thing, img_converted, trans_mat, thing.size());
	imshow("Scaled", img_converted);
}

static void onChangeY(int slider, void* userInput) {
	currentY = -slider;
	Mat trans_mat = (Mat_<double>(2, 3) << 1, 0, currentX, 0, 1, currentY);
	Mat img_converted;
	warpAffine(thing, img_converted, trans_mat, thing.size());
	imshow("Scaled", img_converted);
}

Mat binarization(Mat img) {
	float threshVal;
	threshVal = avgColor(img) * (1 + bK * (sDeviation(img) / bR - 1));
	printf("Thresh: %f", threshVal);
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
	/*if (rRect.size.width < rRect.size.height) {
		printf("True \n");
		blob_angle_deg = 90 + blob_angle_deg;
	}*/


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
	//angle = angle ;
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


void thinningIteration(cv::Mat& im, int iter)
{
	cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);

	for (int i = 1; i < im.rows - 1; i++)
	{
		for (int j = 1; j < im.cols - 1; j++)
		{
			uchar p2 = im.at<uchar>(i - 1, j);
			uchar p3 = im.at<uchar>(i - 1, j + 1);
			uchar p4 = im.at<uchar>(i, j + 1);
			uchar p5 = im.at<uchar>(i + 1, j + 1);
			uchar p6 = im.at<uchar>(i + 1, j);
			uchar p7 = im.at<uchar>(i + 1, j - 1);
			uchar p8 = im.at<uchar>(i, j - 1);
			uchar p9 = im.at<uchar>(i - 1, j - 1);

			int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
				(p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
				(p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
				(p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
			int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
			int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
			int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

			if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
				marker.at<uchar>(i, j) = 1;
		}
	}

	im &= ~marker;
}

/**
* Function for thinning the given binary image
*
* @param  im  Binary image with range = 0-255
// source :https://web.archive.org/web/20160322113207/http://opencv-code.com/quick-tips/implementation-of-thinning-algorithm-in-opencv/
*/
void thinning(cv::Mat& im)
{
	im /= 255;

	cv::Mat prev = cv::Mat::zeros(im.size(), CV_8UC1);
	cv::Mat diff;

	do {
		thinningIteration(im, 0);
		thinningIteration(im, 1);
		cv::absdiff(im, prev, diff);
		im.copyTo(prev);
	} while (cv::countNonZero(diff) > 0);

	im *= 255;
}

//void removeSegments(cv::Mat& im) {
//	Mat labelImage(im.size(), CV_32S);
//	//int labels = connectedComponentsWithStats(im, CC_STAT_AREA, 8);
//	//connectedComponentsWithStats();
//	printf("%d\n", labels);
//}

void removeJunctions(cv::Mat& im) {
	std::vector<vector<int>> junctions;
	for (int i = 1; i < im.rows - 1; i++)
	{
		for (int j = 1; j < im.cols - 1; j++)
		{
			uchar p2 = im.at<uchar>(i - 1, j);
			uchar p3 = im.at<uchar>(i - 1, j + 1);
			uchar p4 = im.at<uchar>(i, j + 1);
			uchar p5 = im.at<uchar>(i + 1, j + 1);
			uchar p6 = im.at<uchar>(i + 1, j);
			uchar p7 = im.at<uchar>(i + 1, j - 1);
			uchar p8 = im.at<uchar>(i, j - 1);
			uchar p9 = im.at<uchar>(i - 1, j - 1);

			int totalNeighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;

			if (totalNeighbors >= 3) {
				// mark the point down for future reference (store in array)
				junctions.push_back({ i,j });
				// remove point
				im.at<uchar>(i, j) = 0;
			}
		}
	}
	for (const std::vector<int> v : junctions) {
		for (int y : v) {
			//printf("%d",y);
		}
	}
}

void pruning(cv::Mat& im) {
	//convert 255 values to 1, 0 values stay at 0
	im /= 255;

	cv::Mat prev = cv::Mat::zeros(im.size(), CV_8UC1);
	cv::Mat diff;

	removeJunctions(im);
	//removeSegments(im);

	// back to 255 to dislay
	im *= 255;

}

//Mat dilate(Mat src) {
//	int dilation_type
//	int dilation_elem = 0;
//	int dilation_size = 1;
//	if (dilation_elem == 0) { dilation_type = MORPH_RECT; }
//	else if (dilation_elem == 1) { dilation_type = MORPH_CROSS; }
//	else if (dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }
//	Mat dilation_dst;
//
//	Mat element = getStructuringElement(dilation_type,
//		Size(2 * dilation_size + 1, 2 * dilation_size + 1),
//		Point(dilation_size, dilation_size));
//	/// Apply the dilation operation
//	dilate(src, dilation_dst, element);
//	dilation_dst = binarization(dilation_dst);
//	dilation_dst = binarization(dilation_dst);
//	return dilation_dst;
//}

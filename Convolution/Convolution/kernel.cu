#if 1

/*
Credit and thanks to https://github.com/Teknoman117/cuda/blob/master/imgproc_example/main.cu
*/

#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h>

#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"

////// CONSTANTS
#define TIME_GPU 0
#define FRAME_RATIO (1.0)
#define OBJ_TEMPLATE_WIDTH (40 * FRAME_RATIO)
#define OBJ_TEMPLATE_HEIGHT (40 * FRAME_RATIO)
#define OBJ_H_RANGE (15)
#define OBJ_S_RANGE (100)
#define OBJ_V_RANGE (100)

////// GPU CONSTANTS
__constant__ float convolutionKernelStore[256];
__constant__ unsigned char structuringElementStore[256];

////// PROTOTYPES
unsigned char *createImageBuffer(unsigned int bytes, unsigned char **devicePtr);
void manualCopy(unsigned char *src, int srcW, int srcH, int startX, int startY, unsigned char *dest, int destW, int destH);
cv::Point centerOfMass(unsigned char *src, int width, int height, float outlierDist, int maxPoints);
void findHSVColor(cv::Mat &src, unsigned char *mask, int width, int height, cv::Point objCenter, int objW, int objH, cv::Vec3b &lower_hsv, cv::Vec3b &upper_hsv);
void handleKeypress(int keypress, cv::Mat frame);

////// GPU PROTOTYPES
void invertBIWrapper(dim3 blocks, dim3 threads, unsigned char *src, int width, int height, unsigned char *dest);
__global__ void invertBI(unsigned char *src, int width, int height, unsigned char *dest);

void logicalAndWrapper(dim3 blocks, dim3 threads, unsigned char *src1, unsigned char *src2, unsigned char *dest, int width, int height);
__global__ void logicalAnd(unsigned char *src1, unsigned char *src2, unsigned char *dest, int width, int height);

void convolveWrapper(dim3 blocks, dim3 threads, unsigned char *src, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *dest);
__global__ void convolve(unsigned char *src, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *dest);

void subtractImagesWrapper(dim3 blocks, dim3 threads, unsigned char *img1, unsigned char *img2, int width, int height, float threshold, unsigned char *dest);
__global__ void subtractImages(unsigned char *img1, unsigned char *img2, int width, int height, float threshold, unsigned char *dest);

void erodeFilterWrapper(dim3 blocks, dim3 threads, unsigned char *src, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *dest);
__global__ void erodeFilter(unsigned char *src, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *dest);

void erodeTemplateFilterWrapper(dim3 blocks, dim3 threads, unsigned char *src, int width, int height, int paddingX, int paddingY, unsigned char *objTemplate, int tWidth, int tHeight, unsigned char *dest);
__global__ void erodeTemplateFilter(unsigned char *src, int width, int height, int paddingX, int paddingY, unsigned char *objTemplate, int tWidth, int tHeight, unsigned char *dest);

void dilateFilterWrapper(dim3 blocks, dim3 threads, unsigned char *src, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *dest);
__global__ void dilateFilter(unsigned char *src, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *dest);

////// ENUMS
enum Operations {
	Normal,
	Greyscale,
	Subtraction,
	Background,
	Tracking
}activeOperation;

////// GLOBAL VARIABLES
float threshold = 20;
int backgroundSet = 0;
int recording = 0;
int videoName = 1;
cv::VideoWriter oVideoWriter;
cv::VideoCapture camera_front(0);
cv::VideoCapture camera_back(1);
cv::VideoCapture camera_usb(2);
cv::VideoCapture activeCamera = camera_front;





#if TIME_GPU
cudaEvent_t start, stop;
#endif

int main() {
	
  if (!camera_front.isOpened()) {
		std::cout << "Camera 0 not opened" << std::endl;
		activeCamera = camera_back;
	}
	if (!camera_back.isOpened()) {
		std::cout << "Camera 1 not opened" << std::endl;
	}
	if (!camera_usb.isOpened()) {
		std::cout << "Camera 2 not opened" << std::endl;
	}
  
	////// CONVOLUTION KERNELS
	size_t convolutionKernelStoreEndOffset = 0;

	const float gaussian5x5Kernel[25] =
	{
		2.f / 159.f,  4.f / 159.f,  5.f / 159.f,  4.f / 159.f, 2.f / 159.f,
		4.f / 159.f,  9.f / 159.f, 12.f / 159.f,  9.f / 159.f, 4.f / 159.f,
		5.f / 159.f, 12.f / 159.f, 15.f / 159.f, 12.f / 159.f, 5.f / 159.f,
		4.f / 159.f,  9.f / 159.f, 12.f / 159.f,  9.f / 159.f, 4.f / 159.f,
		2.f / 159.f,  4.f / 159.f,  5.f / 159.f,  4.f / 159.f, 2.f / 159.f,
	};
	const size_t gaussian5x5KernelOffset = convolutionKernelStoreEndOffset;
	cudaMemcpyToSymbol(convolutionKernelStore, gaussian5x5Kernel, sizeof(gaussian5x5Kernel), gaussian5x5KernelOffset * sizeof(float));
	convolutionKernelStoreEndOffset += sizeof(gaussian5x5Kernel) / sizeof(float);

	const float emboss3x3Kernel[9] =
	{
		1.f, 2.f, 1.f,
		0.f, 0.f, 0.f,
		-1.f, -2.f, -1.f,
		/*-1.f, -1.f, -1.f,
		-1.f, 9.f, -1.f,
		-1.f, -1.f, -1.f,*/
		/*0.f, -1.f, 0.f,
		-1.f, 5.f, -1.f,
		0.f, -1.f, 0.f,*/
	};
	const size_t emboss3x3KernelOffset = convolutionKernelStoreEndOffset;
	cudaMemcpyToSymbol(convolutionKernelStore, emboss3x3Kernel, sizeof(emboss3x3Kernel), emboss3x3KernelOffset * sizeof(float));
	convolutionKernelStoreEndOffset += sizeof(emboss3x3Kernel) / sizeof(float);

	const float outline3x3Kernel[9] =
	{
		-1.f, -1.f, -1.f,
		-1.f, 8.f, -1.f,
		-1.f, -1.f, -1.f,
	};
	const size_t outline3x3KernelOffset = convolutionKernelStoreEndOffset;
	cudaMemcpyToSymbol(convolutionKernelStore, outline3x3Kernel, sizeof(outline3x3Kernel), outline3x3KernelOffset * sizeof(float));
	convolutionKernelStoreEndOffset += sizeof(outline3x3Kernel) / sizeof(float);

	const float leftSobel3x3Kernel[9] =
	{
		1.f, 0.f, -1.f,
		2.f, 0.f, -2.f,
		1.f, 0.f, -1.f,
	};
	const size_t leftSobel3x3KernelOffset = convolutionKernelStoreEndOffset;
	cudaMemcpyToSymbol(convolutionKernelStore, leftSobel3x3Kernel, sizeof(leftSobel3x3Kernel), leftSobel3x3KernelOffset * sizeof(float));
	convolutionKernelStoreEndOffset += sizeof(leftSobel3x3Kernel) / sizeof(float);

	const float topSobel3x3Kernel[9] =
	{
		1.f, 2.f, 1.f,
		0.f, 0.f, 0.f,
		-1.f, -2.f, -1.f,
	};
	const size_t topSobel3x3KernelOffset = convolutionKernelStoreEndOffset;
	cudaMemcpyToSymbol(convolutionKernelStore, topSobel3x3Kernel, sizeof(topSobel3x3Kernel), topSobel3x3KernelOffset * sizeof(float));
	convolutionKernelStoreEndOffset += sizeof(topSobel3x3Kernel) / sizeof(float);

	////// STRUCTURING ELEMENTS
	size_t structuringElementStoreEndOffset = 0;
	const unsigned char binaryCircle3x3[9] =
	{
		0, 1, 0,
		1, 1, 1,
		0, 1, 0
	};
	const size_t binaryCircle3x3Offset = structuringElementStoreEndOffset;
	cudaMemcpyToSymbol(structuringElementStore, binaryCircle3x3, sizeof(binaryCircle3x3), binaryCircle3x3Offset * sizeof(unsigned char));
	structuringElementStoreEndOffset += sizeof(binaryCircle3x3) / sizeof(unsigned char);

	const unsigned char binaryCircle5x5[25] =
	{
		0, 1, 1, 1, 0,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		0, 1, 1, 1, 0
	};
	const size_t binaryCircle5x5Offset = structuringElementStoreEndOffset;
	cudaMemcpyToSymbol(structuringElementStore, binaryCircle5x5, sizeof(binaryCircle5x5), binaryCircle5x5Offset * sizeof(unsigned char));
	structuringElementStoreEndOffset += sizeof(binaryCircle5x5) / sizeof(unsigned char);

	const unsigned char binaryCircle7x7[49] =
	{
		0, 0, 1, 1, 1, 0, 0,
		0, 1, 1, 1, 1, 1, 0,
		1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1,
		0, 1, 1, 1, 1, 1, 0,
		0, 0, 1, 1, 1, 0, 0,
	};
	const size_t binaryCircle7x7Offset = structuringElementStoreEndOffset;
	cudaMemcpyToSymbol(structuringElementStore, binaryCircle7x7, sizeof(binaryCircle7x7), binaryCircle7x7Offset * sizeof(unsigned char));
	structuringElementStoreEndOffset += sizeof(binaryCircle7x7) / sizeof(unsigned char);

	////// VARIABLES
	int frameState = 1;
	int keypress, keypressCur;
	
	// object hsv thresholds
	// blue ball
	cv::Vec3b lower_hsv = { 88, 109, 0 };
	cv::Vec3b upper_hsv = { 118, 255, 199 };
	// red ball
//	cv::Vec3b lower_hsv = { 162, 44, 84 };
//	cv::Vec3b upper_hsv = { 192, 244, 255 };
	// tennis ball
//	lower_hsv - [25, 123, 161]
//	upper_hsv - [29, 154, 212]

	// old blue ball
	//lower_hsv - [98, 80, 35]
	//upper_hsv - [107, 205, 153]
	//lower_hsv - [99, 86, 30]
	//upper_hsv - [107, 205, 147]
	//lower_hsv - [98, 118, 49]
	//upper_hsv - [106, 201, 137]
	//lower_hsv - [96, 48, 43]
	//upper_hsv - [105, 219, 137]
	// blob detector parameters
	cv::SimpleBlobDetector::Params params;
	// params.filterByArea = true;
	// params.minArea = 1;
	// params.maxArea = 300;
	params.filterByColor = true;
	params.blobColor = 255;
	cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
	std::vector<cv::KeyPoint> keypoints;

#if TIME_GPU
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
#endif

	// camera frame
	cv::Mat frame;
	cv::Mat frameOrig;

	// grab the first frame
	activeCamera >> frameOrig;
	cv::resize(frameOrig, frame, cv::Size(frameOrig.size().width * FRAME_RATIO, frameOrig.size().height * FRAME_RATIO));

	// image buffers
	unsigned char *greyscaleDataDevice1,
		*greyscaleDataDevice2,
		*backgroundGreyscaleDataDevice,
		*backgroundGreyscaleBlurredDataDevice,
		*thresholdImageDataDevice,
		*buffer1DataDevice,
		*buffer2DataDevice,
		*displayDataDevice;
	cv::Mat greyscale1(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &greyscaleDataDevice1));
	cv::Mat greyscale2(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &greyscaleDataDevice2));
	cv::Mat backgroundGreyscale(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &backgroundGreyscaleDataDevice));
	cv::Mat backgroundGreyscaleBlurred(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &backgroundGreyscaleBlurredDataDevice));
	cv::Mat thresholdImage(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &thresholdImageDataDevice));
	cv::Mat buffer1(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &buffer1DataDevice));
	cv::Mat buffer2(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &buffer2DataDevice));
	cv::Mat display(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &displayDataDevice));
	cv::Mat background, backgroundOrig,
		hsvBackground, hsvBackgroundOrig,
		greyscale,
		greyscalePrev,
		hsvImage,
		hsvImagePrev,
		hsvImageSub;
	// object template setup
	cv::Size objTemplateSize((OBJ_TEMPLATE_WIDTH * 2) + 1, (OBJ_TEMPLATE_HEIGHT * 2) + 1);
	unsigned char *objTemplate1DataDevice, *objTemplate2DataDevice, *objTemplate3DataDevice, *objTemplate4DataDevice;
	cv::Mat objTemplate1(objTemplateSize, CV_8U, createImageBuffer(objTemplateSize.width * objTemplateSize.height, &objTemplate1DataDevice));
	cv::Mat objTemplate2(objTemplateSize, CV_8U, createImageBuffer(objTemplateSize.width * objTemplateSize.height, &objTemplate2DataDevice));
	cv::Mat objTemplate3(objTemplateSize, CV_8U, createImageBuffer(objTemplateSize.width * objTemplateSize.height, &objTemplate3DataDevice));
	cv::Mat objTemplate4(objTemplateSize, CV_8U, createImageBuffer(objTemplateSize.width * objTemplateSize.height, &objTemplate4DataDevice));

	// GPU settings
	dim3 cblocks(frame.size().width / 16, frame.size().height / 16);
	dim3 cthreads(16, 16);
	dim3 pblocks(frame.size().width * frame.size().height / 256);
	dim3 pthreads(256, 1);

	// object point
	cv::Point objCenter, objCenterPrev;

	// main loop
	while (1) {
		// grab the camera frame
		activeCamera >> frameOrig;
		cv::resize(frameOrig, frame, cv::Size(frameOrig.size().width * FRAME_RATIO, frameOrig.size().height * FRAME_RATIO));

		// convert color spaces
		if (frameState == 1) {
			cv::cvtColor(frame, greyscale1, CV_BGR2GRAY);
			greyscalePrev = greyscale2;
			greyscale = greyscale1;
			frameState = 2;
		}
		else {
			cv::cvtColor(frame, greyscale2, CV_BGR2GRAY);
			greyscalePrev = greyscale1;
			greyscale = greyscale2;
			frameState = 1;
		}
		// convert to HSV
		hsvImagePrev = hsvImage;
		cv::cvtColor(frame, hsvImage, CV_BGR2HSV);

		// state
		{
			switch (activeOperation) {
			case Normal:
				display = frame;
				break;
			case Greyscale:
				display = greyscale;
				break;
			case Subtraction:
				subtractImagesWrapper(cblocks, cthreads, greyscale.data, greyscalePrev.data, frame.size().width, frame.size().height, threshold, buffer1.data);
				display = buffer1;
				break;
			case Background:
				if (backgroundSet) {
					// blur
					//convolveWrapper(cblocks, cthreads, greyscale.data, frame.size().width, frame.size().height, 0, 0, gaussian5x5KernelOffset, 5, 5, buffer1.data);
					// background subtraction
					//subtractImagesWrapper(cblocks, cthreads, buffer1.data, backgroundGreyscaleBlurred.data, frame.size().width, frame.size().height, threshold, buffer2.data);
					cv::absdiff(hsvImage, hsvBackground, hsvImageSub);
					cv::inRange(hsvImageSub, cv::Vec3b(threshold, threshold / 3, 10), cv::Vec3b(255, 255, 255), thresholdImage);
					// open to remove noise
					erodeFilterWrapper(cblocks, cthreads, thresholdImage.data, frame.size().width, frame.size().height, 0, 0, binaryCircle5x5Offset, 5, 5, buffer1.data);
					dilateFilterWrapper(cblocks, cthreads, buffer1.data, frame.size().width, frame.size().height, 0, 0, binaryCircle5x5Offset, 5, 5, buffer2.data);

					display = buffer2;

					// blob detector
					detector->detect(display, keypoints);
					// ignore if multiple blobs found
					if (keypoints.size() > 1) {
						std::cout << "more than one keypoint found" << std::endl;
					}
					else if (keypoints.size() == 1) {
						std::cout << "just one keypoint found - " << keypoints[0].pt << std::endl;
						// grab the center
						cv::Point centerPoint = keypoints[0].pt;

						// pick bounds for blob template
						int left = floor(centerPoint.x) - OBJ_TEMPLATE_WIDTH;
						if (left < 0) left = 0;
						int right = floor(centerPoint.x) + OBJ_TEMPLATE_WIDTH;
						if (right > frame.size().width - 1) right = frame.size().width - 1;
						int top = floor(centerPoint.y) - OBJ_TEMPLATE_HEIGHT;
						if (top < 0) top = 0;
						int bottom = floor(centerPoint.y) + OBJ_TEMPLATE_HEIGHT;
						if (bottom > frame.size().height - 1) bottom = frame.size().height - 1;
						// build the template
						/*cv::Mat part(
							display,
							cv::Range(top, bottom),
							cv::Range(left, right));

            objTemplate1 = part;*/
						manualCopy(display.data, display.size().width, display.size().height, left, top, objTemplate1.data, objTemplate1.size().width, objTemplate1.size().height);
						//cudaMemcpyToSymbol(structuringElementStore, objTemplate1.data, sizeof(objTemplate1.data), objTemplate1Offset * sizeof(unsigned char));
						cv::imshow("Template Buffer1", objTemplate1);
						/*// build the donut template
						// dilate to get bigger object
						dilateFilterWrapper(cblocks, cthreads, objTemplate1.data, objTemplate1.size().width, objTemplate1.size().height, 0, 0, binaryCircle5x5Offset, 5, 5, objTemplate2.data);
						cv::imshow("Template Buffer dilate", objTemplate2);
						dilateFilterWrapper(cblocks, cthreads, objTemplate2.data, objTemplate2.size().width, objTemplate2.size().height, 0, 0, binaryCircle5x5Offset, 5, 5, objTemplate3.data);
						cv::imshow("Template Buffer dilate2", objTemplate3);
						// invert the original object and AND with bigger to get donut
						invertBIWrapper(cblocks, cthreads, objTemplate1.data, objTemplate1.size().width, objTemplate1.size().height, objTemplate2.data);
						cv::imshow("Template Buffer1 inverted", objTemplate2);
						cv::bitwise_and(objTemplate2, objTemplate3, objTemplate4);
						logicalAndWrapper(cblocks, cthreads, objTemplate2.data, objTemplate3.data, objTemplate4.data, objTemplate2.size().width, objTemplate2.size().height);
						cv::imshow("Template Buffer anded", objTemplate4);
						// copy the template into the structuring element store
						//cudaMemcpyToSymbol(structuringElementStore, objTemplate4.data, sizeof(objTemplate4.data), objTemplate2Offset * sizeof(unsigned char));*/

						// find the object's color
						//cv::Vec3b colorOfObj = hsvImage.at<cv::Vec3b>(centerPoint);
						//lower_hsv = cv::Vec3b(max(colorOfObj[0] - OBJ_H_RANGE, 0), max(colorOfObj[1] - OBJ_S_RANGE, 0), max(colorOfObj[2] - OBJ_V_RANGE, 0));
						//upper_hsv = cv::Vec3b(min(colorOfObj[0] + OBJ_H_RANGE, 255), min(colorOfObj[1] + OBJ_S_RANGE, 255), min(colorOfObj[2] + OBJ_V_RANGE, 255));
						findHSVColor(hsvImage, display.data, frame.size().width, frame.size().height, centerPoint, OBJ_TEMPLATE_WIDTH, OBJ_TEMPLATE_HEIGHT, lower_hsv, upper_hsv);
						std::cout << "lower_hsv - " << lower_hsv << std::endl;
						std::cout << "upper_hsv - " << upper_hsv << std::endl;

						cv::circle(display, keypoints[0].pt, 40, cv::Scalar(255, 0, 0), 2);
					}
					else {
						std::cout << "nothing found - " << std::endl;
					}
				}
				else {
					std::cout << "Take Background Picture by pressing 'p' " << std::endl;
					while (1) if (cv::waitKey(1) == 112) break;
					activeCamera >> backgroundOrig;
					cv::resize(backgroundOrig, background, cv::Size(backgroundOrig.size().width * FRAME_RATIO, backgroundOrig.size().height * FRAME_RATIO));
					cv::cvtColor(background, backgroundGreyscale, CV_BGR2GRAY);
					cv::cvtColor(background, hsvBackground, CV_BGR2HSV);
					convolveWrapper(cblocks, cthreads, backgroundGreyscale.data, frame.size().width, frame.size().height, 0, 0, gaussian5x5KernelOffset, 5, 5, backgroundGreyscaleBlurred.data);
					backgroundSet = 1;
				}
				break;
			case Tracking:
			{
				// threshold on object color
				cv::inRange(hsvImage, lower_hsv, upper_hsv, thresholdImage);
				cv::imshow("threshold", thresholdImage);
				// open the image to reduce noise
				erodeFilterWrapper(cblocks, cthreads, thresholdImage.data, frame.size().width, frame.size().height, 0, 0, binaryCircle5x5Offset, 5, 5, buffer1.data);
				dilateFilterWrapper(cblocks, cthreads, buffer1.data, frame.size().width, frame.size().height, 0, 0, binaryCircle5x5Offset, 5, 5, buffer2.data);
				dilateFilterWrapper(cblocks, cthreads, buffer2.data, frame.size().width, frame.size().height, 0, 0, binaryCircle5x5Offset, 5, 5, buffer1.data);
				cv::imshow("dilate, erode", buffer1);
				// erode a whole bunch of times to reduce the object size
				//erodeFilterWrapper(cblocks, cthreads, buffer1.data, frame.size().width, frame.size().height, 0, 0, binaryCircle5x5Offset, 5, 5, buffer2.data);
				//erodeFilterWrapper(cblocks, cthreads, buffer2.data, frame.size().width, frame.size().height, 0, 0, binaryCircle5x5Offset, 5, 5, buffer1.data);
				//erodeFilterWrapper(cblocks, cthreads, buffer1.data, frame.size().width, frame.size().height, 0, 0, binaryCircle5x5Offset, 5, 5, buffer2.data);
				//erodeFilterWrapper(cblocks, cthreads, buffer2.data, frame.size().width, frame.size().height, 0, 0, binaryCircle5x5Offset, 5, 5, buffer1.data);
				//erodeFilterWrapper(cblocks, cthreads, buffer1.data, frame.size().width, frame.size().height, 0, 0, binaryCircle5x5Offset, 5, 5, buffer2.data);
				//erodeFilterWrapper(cblocks, cthreads, buffer2.data, frame.size().width, frame.size().height, 0, 0, binaryCircle5x5Offset, 5, 5, buffer1.data);
				// erode by object template
				//erodeTemplateFilterWrapper(cblocks, cthreads, buffer2.data, frame.size().width, frame.size().height, 100, 100, objTemplate1.data, objTemplate1.size().width, objTemplate1.size().height, buffer1.data);
				display = buffer1;

				// find the center of mass
				objCenterPrev = objCenter;
				objCenter = centerOfMass(buffer1.data, frame.size().width, frame.size().height, OBJ_TEMPLATE_HEIGHT, (OBJ_TEMPLATE_WIDTH * OBJ_TEMPLATE_HEIGHT) / 2);
				if (objCenter.x < 0)
				{
					objCenter = objCenterPrev;
				}
				//memset(buffer1.data, 0, buffer1.size().width*buffer1.size().height);
				cv::circle(frame, objCenter, OBJ_TEMPLATE_WIDTH, cv::Scalar(150, 150, 50), 3);
				display = frame;

				break;
			}
			default:
				break;
			}
		}

		// record to video
		if (recording)
			oVideoWriter.write(display);


		if (display.size().height > 0) {
			cv::namedWindow("Convolution", cv::WINDOW_AUTOSIZE);
			cv::imshow("Convolution", display);
		}

		// handle keypresses
		keypressCur = cv::waitKey(1);
		if (keypressCur < 255) {
			keypress = keypressCur;
			handleKeypress(keypress, frame);
		}
		if (keypress == 27) break;
	}

	// free GPU space
	cudaFreeHost(greyscale1.data);
	cudaFreeHost(greyscale2.data);
	cudaFreeHost(backgroundGreyscale.data);
	cudaFreeHost(backgroundGreyscaleBlurred.data);
	cudaFreeHost(thresholdImage.data);
	cudaFreeHost(objTemplate1.data);
	cudaFreeHost(objTemplate2.data);
	cudaFreeHost(objTemplate3.data);
	cudaFreeHost(objTemplate4.data);
	cudaFreeHost(buffer1.data);
	cudaFreeHost(buffer2.data);
	cudaFreeHost(display.data);
	// free others
	oVideoWriter.release();

	return 0;
}


void manualCopy(unsigned char *src, int srcW, int srcH, int startX, int startY, unsigned char *dest, int destW, int destH)
{
	for (int destY = 0; destY < destH; destY++)
	{
		for (int destX = 0; destX < destW; destX++)
		{
			int srcX = startX + destX;
			int srcY = startY + destY;
			if (srcX < srcW && srcY < srcH)
			{
				dest[(destY*destW) + destX] = src[(srcY*srcW) + srcX];
			}
			else
			{
				dest[(destY*destW) + destX] = 0;
			}
		}
	}
}


cv::Point centerOfMass(unsigned char *src, int width, int height, float outlierDist, int maxPoints)
{
	// variables
	std::vector<cv::Point> points; // points that are on
	float xi = 0.0, yi = 0.0, xf = 0.0, yf = 0.0; // initial and final center of mass coordinates
	int pointsInRange; // points contained within the outlier region
	// build point arrays
	for (int y = 0, pIndex = 0; y < height && pIndex < maxPoints; y++)
	{
		for (int x = 0; x < width && pIndex < maxPoints; x++)
		{
			if (src[(y*width) + x] > 0)
			{
				cv::Point myPoint(x, y);
				points.push_back(myPoint);
			}
		}
	}

	if (points.size() > 0)
	{
		// find initial center of mass
		for (int i = 0; i < points.size(); i++)
		{
			cv::Point p = points[i];
			xi += p.x;
			yi += p.y;
		}
		xi /= points.size();
		yi /= points.size();

		// find new center of mass with outlier
		pointsInRange = 0;
		for (int i = 0; i < points.size(); i++)
		{
			cv::Point p = points[i];
			if (abs(p.x - xi) < outlierDist && abs(p.y - yi) < outlierDist)
			{
				xf += p.x;
				yf += p.y;
				pointsInRange++;
			}
		}
		xf /= pointsInRange;
		yf /= pointsInRange;

		cv::Point centerPoint(xf, yf);
		return centerPoint;
	}

	cv::Point centerPoint(-1.0, -1.0);
	return centerPoint;
}


void findHSVColor(cv::Mat &src, unsigned char *mask, int width, int height, cv::Point objCenter, int objW, int objH, cv::Vec3b &lower_hsv, cv::Vec3b &upper_hsv)
{
	// find average, min, max
	float fraction = 0.9;
	int numPoints = 0;
	double avgH = 0, avgS = 0, avgV = 0;
	int minH = 255, minS = 255, minV = 255;
	int maxH = 0, maxS = 0, maxV = 0;
	for (int y = objCenter.y; y < (objCenter.y + objH) && y < height; y++)
	{
		for (int x = objCenter.x; x < (objCenter.x + objW) && x < width; x++)
		{
			cv::Point p(x, y);
			int pos = (y*width) + x;
			if (mask[pos] > 0)
			{
				numPoints++;
				cv::Vec3b c = src.at<cv::Vec3b>(p);
				minH = min(minH, c[0]);
				minS = min(minS, c[1]);
				minV = min(minV, c[2]);
				maxH = max(maxH, c[0]);
				maxS = max(maxS, c[1]);
				maxV = max(maxV, c[2]);
				avgH += c[0];
				avgS += c[1];
				avgV += c[2];
			}
		}
	}
	avgH /= numPoints;
	avgS /= numPoints;
	avgV /= numPoints;
	minH = avgH - (1.2 * (avgH - minH));
	minS = avgS - (0.675 * (avgS - minS)) + 20;
	minV = avgV - (1.25 * (avgV - minV));
	maxH = avgH + (1.2 * (maxH - avgH));
	maxS = avgS + (0.675 * (maxS - avgS)) + 20;
	maxV = avgV + (1.25 * (maxV - avgV));

	lower_hsv[0] = minH;
	lower_hsv[1] = minS;
	lower_hsv[2] = minV;
	upper_hsv[0] = maxH;
	upper_hsv[1] = maxS;
	upper_hsv[2] = maxV;
}


void handleKeypress(int keypress, cv::Mat frame) {
	switch (keypress) {
	case 97: /* a */
		activeOperation = Normal;
		break;
	case 115: /* s */
		activeOperation = Greyscale;
		break;
	case 100: /* d */
		activeOperation = Subtraction;
		break;
	case 102: /* f */
		activeOperation = Background;
		break;
	case 103: /* g */
		activeOperation = Tracking;
		break;
	case 104: /* h */
		break;

	case 112: /* p */
		backgroundSet = 0;
		break;


	case 113: /* q */
		activeCamera = camera_front;
		break;
	case 119: /* w */
		activeCamera = camera_back;
		break;
	case 101: /* e */
		activeCamera = camera_usb;
		break;

	case 116: /* t */
		threshold += 5;
		std::cout << "\nThreshold = " << threshold << "\n" << std::endl;
		break;
	case 121: /* y */
		threshold -= 5;
		std::cout << "\nThreshold = " << threshold << "\n" << std::endl;
		break;

	case 109: /* m */
		oVideoWriter.open(std::to_string(videoName) + ".avi", CV_FOURCC('I', 'Y', 'U', 'V'), 20, frame.size(), true);
		recording = 1;
		videoName++;
		break;
	case 110: /* n */
		recording = 0;
		break;
	default:
		break;
	}

}


unsigned char *createImageBuffer(unsigned int bytes, unsigned char **devicePtr) {

	unsigned char *ptr = NULL;
	cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaHostAlloc(&ptr, bytes, cudaHostAllocMapped);
	cudaHostGetDevicePointer(devicePtr, ptr, 0);
	return ptr;
}


void invertBIWrapper(dim3 blocks, dim3 threads, unsigned char *src, int width, int height, unsigned char *dest)
{
#if TIME_GPU
	cudaEventRecord(start);
#endif

	invertBI << <blocks, threads >> >(src, width, height, dest);
	cudaDeviceSynchronize();

#if TIME_GPU
	cudaEventRecord(stop);
	float ms = 0.0f;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	std::cout << "Elapsed GPU time: " << ms << " milliseconds" << std::endl;
#endif
}
__global__ void invertBI(unsigned char *src, int width, int height, unsigned char *dest)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int i = (y * width) + x;

	dest[i] = ( src[i] > 0 ) ? 0 : 255;
}


void logicalAndWrapper(dim3 blocks, dim3 threads, unsigned char *src1, unsigned char *src2, unsigned char *dest, int width, int height) {
#if TIME_GPU
	cudaEventRecord(start);
#endif

	logicalAnd << <blocks, threads >> >(src1, src2, dest, width, height);
	cudaDeviceSynchronize();

#if TIME_GPU
	cudaEventRecord(stop);
	float ms = 0.0f;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	std::cout << "Elapsed GPU time: " << ms << " milliseconds" << std::endl;
#endif
}
__global__ void logicalAnd(unsigned char *src1, unsigned char *src2, unsigned char *dest, int width, int height)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int i = (y * width) + x;

	dest[i] = ((src1[i] > 0) && (src2[i] > 0)) ? 255 : 0;
}


void convolveWrapper(dim3 blocks, dim3 threads, unsigned char *src, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *dest) {
#if TIME_GPU
	cudaEventRecord(start);
#endif

	convolve << <blocks, threads >> > (src, width, height, paddingX, paddingY, kOffset, kWidth, kHeight, dest);
	cudaDeviceSynchronize();

#if TIME_GPU
	cudaEventRecord(stop);
	float ms = 0.0f;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	std::cout << "Elapsed GPU time: " << ms << " milliseconds" << std::endl;
#endif
}
__global__ void convolve(unsigned char *src, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *dest) {

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	float sum = 0.0;
	int pWidth = kWidth / 2;
	int pHeight = kHeight / 2;

	// Only execute for valid pixels
	if (x >= pWidth + paddingX &&
		y >= pHeight + paddingY &&
		x < (blockDim.x * gridDim.x) - pWidth - paddingX &&
		y < (blockDim.y * gridDim.y) - pHeight - paddingY)
	{
		for (int j = -pHeight; j <= pHeight; j++)
		{
			for (int i = -pWidth; i <= pWidth; i++)
			{
				// Sample the weight for this location
				int ki = (i + pWidth);
				int kj = (j + pHeight);
				float w = convolutionKernelStore[(kj * kWidth) + ki + kOffset];


				sum += w * float(src[((y + j) * width) + (x + i)]);
			}
		}
	}

	dest[(y * width) + x] = (unsigned char)sum;
}


void subtractImagesWrapper(dim3 blocks, dim3 threads, unsigned char *img1, unsigned char *img2, int width, int height, float threshold, unsigned char *dest) {
#if TIME_GPU
	cudaEventRecord(start);
#endif

	subtractImages << <blocks, threads >> > (img1, img2, width, height, threshold, dest);
	cudaDeviceSynchronize();

#if TIME_GPU
	cudaEventRecord(stop);
	float ms = 0.0f;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	std::cout << "Elapsed GPU time: " << ms << " milliseconds" << std::endl;
#endif
}
__global__ void subtractImages(unsigned char *img1, unsigned char *img2, int width, int height, float threshold, unsigned char *dest) {

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	float pixelDifference = abs(img1[(y * width) + x] - img2[(y * width) + x]);
	if (pixelDifference > threshold) {
		dest[(y * width) + x] = 255;
	}
	else {
		dest[(y * width) + x] = 0;
	}
}


void erodeFilterWrapper(dim3 blocks, dim3 threads, unsigned char *src, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *dest) {
#if TIME_GPU
	cudaEventRecord(start);
#endif

	erodeFilter << <blocks, threads >> > (src, width, height, paddingX, paddingY, kOffset, kWidth, kHeight, dest);
	cudaDeviceSynchronize();

#if TIME_GPU
	cudaEventRecord(stop);
	float ms = 0.0f;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	std::cout << "Elapsed GPU time: " << ms << " milliseconds" << std::endl;
#endif
}
__global__ void erodeFilter(unsigned char *src, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *dest)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	bool erode = false;
	int pWidth = kWidth / 2;
	int pHeight = kHeight / 2;

	// Only execute for valid pixels
	if (x >= pWidth + paddingX &&
		y >= pHeight + paddingY &&
		x < (blockDim.x * gridDim.x) - pWidth - paddingX &&
		y < (blockDim.y * gridDim.y) - pHeight - paddingY)
	{
		for (int j = -pHeight; j <= pHeight && !erode; j++)
		{
			for (int i = -pWidth; i <= pWidth && !erode; i++)
			{
				// Sample the weight for this location
				int ki = (i + pWidth);
				int kj = (j + pHeight);
				if (structuringElementStore[(kj * kWidth) + ki + kOffset] > 0)
				{
					int px = x + i;
					int py = y + j;
					erode = !(src[(py * width) + px] > 0);
				}
			}
		}
		dest[(y * width) + x] = (erode) ? 0 : 255;
	}
}


void erodeTemplateFilterWrapper(dim3 blocks, dim3 threads, unsigned char *src, int width, int height, int paddingX, int paddingY, unsigned char *objTemplate, int tWidth, int tHeight, unsigned char *dest) {
#if TIME_GPU
	cudaEventRecord(start);
#endif

	erodeTemplateFilter << <blocks, threads >> > (src, width, height, paddingX, paddingY, objTemplate, tWidth, tHeight, dest);
	cudaDeviceSynchronize();

#if TIME_GPU
	cudaEventRecord(stop);
	float ms = 0.0f;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	std::cout << "Elapsed GPU time: " << ms << " milliseconds" << std::endl;
#endif
}
__global__ void erodeTemplateFilter(unsigned char *src, int width, int height, int paddingX, int paddingY, unsigned char *objTemplate, int tWidth, int tHeight, unsigned char *dest)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	bool erode = false;
	int pWidth = tWidth / 2;
	int pHeight = tHeight / 2;

	// Only execute for valid pixels
	if (x >= pWidth + paddingX &&
		y >= pHeight + paddingY &&
		x < (blockDim.x * gridDim.x) - pWidth - paddingX &&
		y < (blockDim.y * gridDim.y) - pHeight - paddingY)
	{
		for (int j = -pHeight; j <= pHeight && !erode; j++)
		{
			for (int i = -pWidth; i <= pWidth && !erode; i++)
			{
				// Sample the weight for this location
				int ki = (i + pWidth);
				int kj = (j + pHeight);
				if (objTemplate[(kj * tWidth) + ki] > 0)
				{
					int px = x + i;
					int py = y + j;
					erode = !(src[(py * width) + px] > 0);
				}
			}
		}
		dest[(y * width) + x] = (erode) ? 0 : 255;
	}
}


void dilateFilterWrapper(dim3 blocks, dim3 threads, unsigned char *src, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *dest) {
#if TIME_GPU
	cudaEventRecord(start);
#endif
	memset(dest, 0, width*height);
	dilateFilter << <blocks, threads >> > (src, width, height, paddingX, paddingY, kOffset, kWidth, kHeight, dest);
	cudaDeviceSynchronize();

#if TIME_GPU
	cudaEventRecord(stop);
	float ms = 0.0f;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	std::cout << "Elapsed GPU time: " << ms << " milliseconds" << std::endl;
#endif
}
__global__ void dilateFilter(unsigned char *src, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *dest)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	int pWidth = kWidth / 2;
	int pHeight = kHeight / 2;

	// Only execute for valid pixels
	if (x >= pWidth + paddingX &&
		y >= pHeight + paddingY &&
		x < (blockDim.x * gridDim.x) - pWidth - paddingX &&
		y < (blockDim.y * gridDim.y) - pHeight - paddingY)
	{
		// is this pixel on?
		if (src[(y * width) + x] > 0)
		{
			// Dilate!!!
			for (int j = -pHeight; j <= pHeight; j++)
			{
				for (int i = -pWidth; i <= pWidth; i++)
				{
					int ki = (i + pWidth);
					int kj = (j + pHeight);
					if (structuringElementStore[(kj * kWidth) + ki + kOffset] > 0)
					{
						int px = x + i;
						int py = y + j;
						dest[(py * width) + px] = 255;
					}
				}
			}
		}
	}
}
#endif




/* Convolution Filter Video */
#if 0

/*
Credit and thanks to https://github.com/Teknoman117/cuda/blob/master/imgproc_example/main.cu
*/

#define TIME_GPU 1

#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h>

#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#if TIME_GPU
cudaEvent_t start, stop;
#endif

__constant__ float convolutionKernelStore[256];

unsigned char *createImageBuffer(unsigned int bytes, unsigned char **devicePtr);

void pythagorasWrapper(dim3 blocks, dim3 threads, unsigned char *a, unsigned char *b, unsigned char *c);
__global__ void pythagoras(unsigned char *a, unsigned char *b, unsigned char *c);

void convolveWrapper(dim3 blocks, dim3 threads, unsigned char *src, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *dest);
__global__ void convolve(unsigned char *src, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *dest);

void pythagoras_slow(unsigned char *a, unsigned char *b, unsigned char *c, int width, int height);
void convolve_slow(unsigned char *src, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, const float *kernel, unsigned char *dest);

void handleKeypress();
int keypress;

enum Kernels {
	Normal,
	Greyscale,
	Blurred,
	Embossed,
	Outline,
	Sobel
}activeKernel;
cv::VideoCapture camera_front(0);
cv::VideoCapture camera_back(1);
cv::VideoCapture camera_usb(2);
cv::VideoCapture activeCamera = camera_front;

int activeProcessing = 0; /* 0 = Use the GPU. 1 = Use the CPU */

int main() {

	cv::Mat frame;
	if (!camera_front.isOpened()) {
		std::cout << "Camera 0 not opened" << std::endl;
		activeCamera = camera_back;
	}
	if (!camera_back.isOpened()) {
		std::cout << "Camera 1 not opened" << std::endl;
	}
	if (!camera_usb.isOpened()) {
		std::cout << "Camera 2 not opened" << std::endl;
	}

#if TIME_GPU
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
#endif

	clock_t start_clock, end_clock;
	double cpu_time_used;

	const float gaussian5x5Kernel[25] =
	{
		2.f / 159.f,  4.f / 159.f,  5.f / 159.f,  4.f / 159.f, 2.f / 159.f,
		4.f / 159.f,  9.f / 159.f, 12.f / 159.f,  9.f / 159.f, 4.f / 159.f,
		5.f / 159.f, 12.f / 159.f, 15.f / 159.f, 12.f / 159.f, 5.f / 159.f,
		4.f / 159.f,  9.f / 159.f, 12.f / 159.f,  9.f / 159.f, 4.f / 159.f,
		2.f / 159.f,  4.f / 159.f,  5.f / 159.f,  4.f / 159.f, 2.f / 159.f,
	};
	cudaMemcpyToSymbol(convolutionKernelStore, gaussian5x5Kernel, sizeof(gaussian5x5Kernel), 0);
	const size_t gaussian5x5KernelOffset = 0;

	const float emboss3x3Kernel[9] =
	{
		1.f, 2.f, 1.f,
		0.f, 0.f, 0.f,
		-1.f, -2.f, -1.f,
		/*-1.f, -1.f, -1.f,
		-1.f, 9.f, -1.f,
		-1.f, -1.f, -1.f,*/
		/*0.f, -1.f, 0.f,
		-1.f, 5.f, -1.f,
		0.f, -1.f, 0.f,*/
	};
	cudaMemcpyToSymbol(convolutionKernelStore, emboss3x3Kernel, sizeof(emboss3x3Kernel), sizeof(gaussian5x5Kernel));
	const size_t emboss3x3KernelOffset = sizeof(gaussian5x5Kernel) / sizeof(float);

	const float outline3x3Kernel[9] =
	{
		-1.f, -1.f, -1.f,
		-1.f, 8.f, -1.f,
		-1.f, -1.f, -1.f,
	};
	cudaMemcpyToSymbol(convolutionKernelStore, outline3x3Kernel, sizeof(outline3x3Kernel), sizeof(gaussian5x5Kernel) + sizeof(emboss3x3Kernel));
	const size_t outline3x3KernelOffset = sizeof(emboss3x3Kernel) / sizeof(float) + emboss3x3KernelOffset;

	const float leftSobel3x3Kernel[9] =
	{
		1.f, 0.f, -1.f,
		2.f, 0.f, -2.f,
		1.f, 0.f, -1.f,
	};
	cudaMemcpyToSymbol(convolutionKernelStore, leftSobel3x3Kernel, sizeof(leftSobel3x3Kernel), sizeof(gaussian5x5Kernel) + sizeof(emboss3x3Kernel) + sizeof(outline3x3Kernel));
	const size_t leftSobel3x3KernelOffset = sizeof(outline3x3Kernel) / sizeof(float) + outline3x3KernelOffset;
	const float topSobel3x3Kernel[9] =
	{
		1.f, 2.f, 1.f,
		0.f, 0.f, 0.f,
		-1.f, -2.f, -1.f,
	};
	cudaMemcpyToSymbol(convolutionKernelStore, topSobel3x3Kernel, sizeof(topSobel3x3Kernel), sizeof(gaussian5x5Kernel) + sizeof(emboss3x3Kernel) + sizeof(outline3x3Kernel) + sizeof(leftSobel3x3Kernel));
	const size_t topSobel3x3KernelOffset = sizeof(leftSobel3x3Kernel) / sizeof(float) + leftSobel3x3KernelOffset;

	activeCamera >> frame;
	unsigned char *leftSobelDataDevice, *topSobelDataDevice;
	unsigned char *greyscaleDataDevice, *blurredDataDevice, *embossedDataDevice, *outlineDataDevice, *sobelDataDevice, *displayDataDevice;
	cv::Mat greyscale(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &greyscaleDataDevice));
	cv::Mat blurred(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &blurredDataDevice));
	cv::Mat embossed(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &embossedDataDevice));
	cv::Mat outline(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &outlineDataDevice));
	cv::Mat leftSobel(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &leftSobelDataDevice));
	cv::Mat topSobel(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &topSobelDataDevice));
	cv::Mat sobel(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &sobelDataDevice));
	cv::Mat display(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &displayDataDevice));

	int keypressCur;



	while (1) {
		activeCamera >> frame;
		cv::cvtColor(frame, greyscale, CV_BGR2GRAY);


		if (activeProcessing == 0) {
			dim3 cblocks(frame.size().width / 16, frame.size().height / 16);
			dim3 cthreads(16, 16);
			dim3 pblocks(frame.size().width * frame.size().height / 256);
			dim3 pthreads(256, 1);
			{
				switch (activeKernel) {
				case Normal:
					display = frame;
					break;
				case Greyscale:
					display = greyscale;
					break;
				case Blurred:
					convolveWrapper(cblocks, cthreads, greyscale.data, frame.size().width, frame.size().height, 0, 0, gaussian5x5KernelOffset, 5, 5, blurred.data);
					display = blurred;
					break;
				case Embossed:
					convolveWrapper(cblocks, cthreads, greyscale.data, frame.size().width, frame.size().height, 0, 0, emboss3x3KernelOffset, 3, 3, embossed.data);
					display = embossed;
					break;
				case Outline:
					convolveWrapper(cblocks, cthreads, greyscale.data, frame.size().width, frame.size().height, 0, 0, outline3x3KernelOffset, 3, 3, outline.data);
					display = outline;
					break;
				case Sobel:
					convolveWrapper(cblocks, cthreads, greyscale.data, frame.size().width, frame.size().height, 0, 0, gaussian5x5KernelOffset, 5, 5, blurred.data);
					convolveWrapper(cblocks, cthreads, blurred.data, frame.size().width, frame.size().height, 0, 0, leftSobel3x3KernelOffset, 3, 3, leftSobel.data);
					convolveWrapper(cblocks, cthreads, blurred.data, frame.size().width, frame.size().height, 0, 0, topSobel3x3KernelOffset, 3, 3, topSobel.data);
					pythagorasWrapper(pblocks, pthreads, leftSobel.data, topSobel.data, sobel.data);
					display = sobel;
					break;
				default:
					break;
				}
			}
		}
		else if (activeProcessing == 1) {
			start_clock = clock();
			switch (activeKernel) {
			case Normal:
				display = frame;
				break;
			case Greyscale:
				display = greyscale;
				break;
			case Blurred:
				convolve_slow(greyscale.data, frame.size().width, frame.size().height, 0, 0, gaussian5x5KernelOffset, 5, 5, gaussian5x5Kernel, blurred.data);
				display = blurred;
				break;
			case Embossed:
				convolve_slow(greyscale.data, frame.size().width, frame.size().height, 0, 0, emboss3x3KernelOffset, 3, 3, emboss3x3Kernel, embossed.data);
				display = embossed;
				break;
			case Outline:
				convolve_slow(greyscale.data, frame.size().width, frame.size().height, 0, 0, outline3x3KernelOffset, 3, 3, outline3x3Kernel, outline.data);
				display = outline;
				break;
			case Sobel:
				convolve_slow(greyscale.data, frame.size().width, frame.size().height, 0, 0, gaussian5x5KernelOffset, 5, 5, gaussian5x5Kernel, blurred.data);
				convolve_slow(blurred.data, frame.size().width, frame.size().height, 0, 0, leftSobel3x3KernelOffset, 3, 3, leftSobel3x3Kernel, leftSobel.data);
				convolve_slow(blurred.data, frame.size().width, frame.size().height, 0, 0, topSobel3x3KernelOffset, 3, 3, topSobel3x3Kernel, topSobel.data);
				pythagoras_slow(leftSobel.data, topSobel.data, sobel.data, frame.size().width, frame.size().height);
				display = sobel;
			default:
				break;
			}
			end_clock = clock();
			cpu_time_used = ((double)end_clock - start_clock) / CLOCKS_PER_SEC;
			std::cout << "Elapsed CPU time: " << cpu_time_used * 1000 << " milliseconds" << std::endl;
		}


		if (display.size().height > 0) {
			cv::namedWindow("Convolution", cv::WINDOW_AUTOSIZE);
			cv::imshow("Convolution", display);
		}

		keypressCur = cv::waitKey(1);
		if (keypressCur < 255) {
			keypress = keypressCur;
			handleKeypress();
		}

		if (keypress == 27) break;
	}

	cudaFreeHost(greyscale.data);
	cudaFreeHost(blurred.data);
	cudaFreeHost(embossed.data);
	cudaFreeHost(outline.data);
	cudaFreeHost(leftSobel.data);
	cudaFreeHost(topSobel.data);
	cudaFreeHost(sobel.data);
	cudaFreeHost(display.data);

	return 0;
}


unsigned char *createImageBuffer(unsigned int bytes, unsigned char **devicePtr) {

	unsigned char *ptr = NULL;
	cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaHostAlloc(&ptr, bytes, cudaHostAllocMapped);
	cudaHostGetDevicePointer(devicePtr, ptr, 0);
	return ptr;
}

void pythagorasWrapper(dim3 blocks, dim3 threads, unsigned char *a, unsigned char *b, unsigned char *c) {
#if TIME_GPU
	cudaEventRecord(start);
#endif

	pythagoras << <blocks, threads >> >(a, b, c);
	cudaDeviceSynchronize();

#if TIME_GPU
	cudaEventRecord(stop);
	float ms = 0.0f;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	std::cout << "Elapsed GPU time: " << ms << " milliseconds" << std::endl;
#endif

}
__global__ void pythagoras(unsigned char *a, unsigned char *b, unsigned char *c)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	float af = float(a[idx]);
	float bf = float(b[idx]);

	c[idx] = (unsigned char)sqrtf(af*af + bf*bf);
}


void convolveWrapper(dim3 blocks, dim3 threads, unsigned char *src, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *dest) {
#if TIME_GPU
	cudaEventRecord(start);
#endif

	convolve << <blocks, threads >> > (src, width, height, paddingX, paddingY, kOffset, kWidth, kHeight, dest);
	cudaDeviceSynchronize();

#if TIME_GPU
	cudaEventRecord(stop);
	float ms = 0.0f;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	std::cout << "Elapsed GPU time: " << ms << " milliseconds" << std::endl;
#endif
}
__global__ void convolve(unsigned char *src, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *dest) {

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	float sum = 0.0;
	int pWidth = kWidth / 2;
	int pHeight = kHeight / 2;

	// Only execute for valid pixels
	if (x >= pWidth + paddingX &&
		y >= pHeight + paddingY &&
		x < (blockDim.x * gridDim.x) - pWidth - paddingX &&
		y < (blockDim.y * gridDim.y) - pHeight - paddingY)
	{
		for (int j = -pHeight; j <= pHeight; j++)
		{
			for (int i = -pWidth; i <= pWidth; i++)
			{
				// Sample the weight for this location
				int ki = (i + pWidth);
				int kj = (j + pHeight);
				float w = convolutionKernelStore[(kj * kWidth) + ki + kOffset];


				sum += w * float(src[((y + j) * width) + (x + i)]);
			}
		}
	}

	dest[(y * width) + x] = (unsigned char)sum;
}


void pythagoras_slow(unsigned char *a, unsigned char *b, unsigned char *c, int width, int height)
{
	for (int i = 0; i < width * height; i++) {

		float af = float(a[i]);
		float bf = float(b[i]);
		c[i] = (unsigned char)sqrtf(af*af + bf*bf);
	}

}

void convolve_slow(unsigned char *src, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, const float *kernel, unsigned char *dest) {

	/*for (int y = 0; y < height; y++)
	{
	for (int x = 0; x < width; x++)
	{
	src[(y * width) + x]
	}*/
	for (int k = 0; k < width * height; k++) {
		float sum = 0.0;
		int x = k % width;
		int y = k / width;

		// Only execute for valid pixels
		if (x >= paddingX &&
			y >= paddingY &&
			x < (width)-paddingX &&
			y < (height)-paddingY)
		{
			for (int j = 0; j < kHeight; j++)
			{
				for (int i = 0; i < kWidth; i++)
				{
					// Sample the weight for this location
					float w = kernel[(j * kWidth) + i];

					sum += w * src[((y + j) * width) + (x + i)];

					/*if ((k > 2000) && (k < 2010)) {
					std::cout << "Kernel Index " << (int)(kj * kWidth) + ki << "   val at Kernel Index " << w <<  "    K " << k <<  "   src Index " << ((y + j) * width) + (x + i) << "    Sum " << sum << std::endl;
					}*/
				}
			}
		}

		sum = (sum < 0) ? 0 : sum;
		sum = (sum > 255) ? 255 : sum;
		dest[k] = sum;
		/*if ((k > 2000) && (k < 2010)) {
		std::cout << (int) src[k] << "    " << (int) dest[k] << "    " << std::endl;
		}*/
	}
}

void handleKeypress() {

	switch (keypress) {
	case 97: /* a */
		activeKernel = Normal;
		break;
	case 115: /* s */
		activeKernel = Greyscale;
		break;
	case 100: /* d */
		activeKernel = Blurred;
		break;
	case 102: /* f */
		activeKernel = Embossed;
		break;
	case 103: /* g */
		activeKernel = Outline;
		break;
	case 104: /* h */
		activeKernel = Sobel;
		break;

	case 113: /* q */
		activeCamera = camera_front;
		break;
	case 119: /* w */
		activeCamera = camera_back;
		break;

	case 122: /* z */
		activeProcessing = 0;
		break;
	case 120: /* x */
		activeProcessing = 1;
		break;
	default:
		break;
	}

}
#endif














#if 0
#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include <time.h>
#include <windows.h>


/* Keyword __global__ tells compiler that the this function will execute on the GPU and is callable from the host */
__global__ void VectorAdd(int *a, int *b, int n) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) b[i] = a[i] + b[i];

}

void VectorAdd_slow(int *d, int *e, int n) {

	for (int i = 0; i < n; i++) {
		e[i] = d[i] + e[i];
	}
}

int main() {

	int N = 1 << 20;

	int *a, *b, *d, *e;

	cudaMallocManaged(&a, N * sizeof(int));
	cudaMallocManaged(&b, N * sizeof(int));

	d = (int *)malloc(N * sizeof(int));
	e = (int *)malloc(N * sizeof(int));

	for (int i = 0; i < N; i++) {

		a[i] = i;
		b[i] = i;

		d[i] = i;
		e[i] = i;
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	clock_t start_clock, end_clock;
	double cpu_time_used;

	/*dim3 cblocks(N / 16, N / 16);
	dim3 cthreads(16, 16);*/
	dim3 pblocks(256);
	dim3 pthreads(32);


	cudaEventRecord(start);
	VectorAdd << <pblocks, pthreads >> > (a, b, N); //launch config <<< number of thread blocks, number of threads within each block >>>
													/* Nubmer of threads in block must be multiple of 32 */
	cudaThreadSynchronize();
	cudaEventRecord(stop);
	float ms = 0.0f;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	std::cout << "Elapsed GPU time: " << ms << " milliseconds" << std::endl;


	start_clock = clock();
	VectorAdd_slow(d, e, N);
	end_clock = clock();
	cpu_time_used = ((double)end_clock - start_clock) / CLOCKS_PER_SEC;
	std::cout << "Elapsed CPU time: " << cpu_time_used * 1000 << " milliseconds" << std::endl;


	for (int i = 32; i < 128; i++) {

		std::cout << "b[" << i << "] = " << b[i] << "    e[" << i << "] = " << e[i] << std::endl;
	}

	cudaFree(a);
	cudaFree(b);

	free(d);
	free(e);

	return 0;
}
#endif

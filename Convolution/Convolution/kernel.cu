#if 1

/*
Credit and thanks to https://github.com/Teknoman117/cuda/blob/master/imgproc_example/main.cu
*/

// GPU benchmark control
#define TIME_GPU 0

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

#if TIME_GPU
cudaEvent_t start, stop;
#endif

// convolution kernels
__constant__ float convolutionKernelStore[256];
// structuring elements
__constant__ bool structuringElementStore[10000];

unsigned char *createImageBuffer(unsigned int bytes, unsigned char **devicePtr);

void invertBIWrapper(dim3 blocks, dim3 threads, unsigned char *source, int width, int height, unsigned char *destination);
__global__ void invertBI(unsigned char *source, int width, int height, unsigned char *destination);

void logicalAndWrapper(dim3 blocks, dim3 threads, unsigned char *a, unsigned char *b, unsigned char *c, int width, int height);
__global__ void logicalAnd(unsigned char *a, unsigned char *b, unsigned char *c, int width, int height);

void pythagorasWrapper(dim3 blocks, dim3 threads, unsigned char *a, unsigned char *b, unsigned char *c);
__global__ void pythagoras(unsigned char *a, unsigned char *b, unsigned char *c);

void convolveWrapper(dim3 blocks, dim3 threads, unsigned char *source, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *destination);
__global__ void convolve(unsigned char *source, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *destination);

void subtractImagesWrapper(dim3 blocks, dim3 threads, unsigned char *img1, unsigned char *img2, int width, int height, float threshold, unsigned char *destination);
__global__ void subtractImages(unsigned char *img1, unsigned char *img2, int width, int height, float threshold, unsigned char *destination);

void erodeFilterWrapper(dim3 blocks, dim3 threads, unsigned char *source, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *destination);
__global__ void erodeFilter(unsigned char *source, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *destination);

void dilateFilterWrapper(dim3 blocks, dim3 threads, unsigned char *source, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *destination);
__global__ void dilateFilter(unsigned char *source, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *destination);

void centerOfMass(cv::Point centerPoint, unsigned char *source, int width, int height, int outlierDist, int maxPoints);
void handleKeypress(cv::Mat frame);

int keypress;
int recording, videoName = 1;
cv::VideoWriter oVideoWriter;

enum Operations {
	Normal,
	Greyscale,
	Subtraction,
	Background,
	Tracking
}activeOperation;
cv::VideoCapture camera_front(0);
cv::VideoCapture camera_back(1);
cv::VideoCapture camera_usb(2);
cv::VideoCapture activeCamera = camera_front;

// default image subtraction threshold
float threshold = 20;
int backgroundSet = 0;

int main() {
	if (!camera_front.isOpened()) {
		std::cout << "Front camera not opened" << std::endl;
		activeCamera = camera_back;
	}
	if (!camera_back.isOpened()) {
		std::cout << "Back camera not opened" << std::endl;
	}
	if (!camera_usb.isOpened()) {
		std::cout << "USB camera not opened" << std::endl;
	}

	cv::Mat frame;
	cv::Mat background;

	recording = 0;

#if TIME_GPU
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
#endif

	/// CONVOLUTION KERNELS
	size_t convolutionKernelStoreEndOffset = 0;
	const float gaussian5x5Kernel[25] =
	{
		2.f / 159.f,  4.f / 159.f,  5.f / 159.f,  4.f / 159.f, 2.f / 159.f,
		4.f / 159.f,  9.f / 159.f, 12.f / 159.f,  9.f / 159.f, 4.f / 159.f,
		5.f / 159.f, 12.f / 159.f, 15.f / 159.f, 12.f / 159.f, 5.f / 159.f,
		4.f / 159.f,  9.f / 159.f, 12.f / 159.f,  9.f / 159.f, 4.f / 159.f,
		2.f / 159.f,  4.f / 159.f,  5.f / 159.f,  4.f / 159.f, 2.f / 159.f,
	};
	const size_t gaussian5x5KernelSize = sizeof(gaussian5x5Kernel) / sizeof(gaussian5x5Kernel[0]);
	const size_t gaussian5x5KernelOffset = convolutionKernelStoreEndOffset;
	cudaMemcpyToSymbol(convolutionKernelStore, gaussian5x5Kernel, gaussian5x5KernelSize, gaussian5x5KernelOffset);
	convolutionKernelStoreEndOffset += gaussian5x5KernelSize;

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
	const size_t emboss3x3KernelSize = sizeof(emboss3x3Kernel) / sizeof(emboss3x3Kernel[0]);
	const size_t emboss3x3KernelOffset = convolutionKernelStoreEndOffset;
	cudaMemcpyToSymbol(convolutionKernelStore, emboss3x3Kernel, emboss3x3KernelSize, emboss3x3KernelOffset);
	convolutionKernelStoreEndOffset += emboss3x3KernelSize;

	const float outline3x3Kernel[9] =
	{
		-1.f, -1.f, -1.f,
		-1.f, 8.f, -1.f,
		-1.f, -1.f, -1.f,
	};
	const size_t outline3x3KernelSize = sizeof(outline3x3Kernel) / sizeof(outline3x3Kernel[0]);
	const size_t outline3x3KernelOffset = convolutionKernelStoreEndOffset;
	cudaMemcpyToSymbol(convolutionKernelStore, outline3x3Kernel, outline3x3KernelSize, outline3x3KernelOffset);
	convolutionKernelStoreEndOffset += outline3x3KernelSize;

	const float leftSobel3x3Kernel[9] =
	{
		1.f, 0.f, -1.f,
		2.f, 0.f, -2.f,
		1.f, 0.f, -1.f,
	};
	const size_t leftSobel3x3KernelSize = sizeof(leftSobel3x3Kernel) / sizeof(leftSobel3x3Kernel[0]);
	const size_t leftSobel3x3KernelOffset = convolutionKernelStoreEndOffset;
	cudaMemcpyToSymbol(convolutionKernelStore, leftSobel3x3Kernel, leftSobel3x3KernelSize, leftSobel3x3KernelOffset);
	convolutionKernelStoreEndOffset += leftSobel3x3KernelSize;

	const float topSobel3x3Kernel[9] =
	{
		1.f, 2.f, 1.f,
		0.f, 0.f, 0.f,
		-1.f, -2.f, -1.f,
	};
	const size_t topSobel3x3KernelSize = sizeof(topSobel3x3Kernel) / sizeof(topSobel3x3Kernel[0]);
	const size_t topSobel3x3KernelOffset = convolutionKernelStoreEndOffset;
	cudaMemcpyToSymbol(convolutionKernelStore, topSobel3x3Kernel, topSobel3x3KernelSize, topSobel3x3KernelOffset);
	convolutionKernelStoreEndOffset += topSobel3x3KernelSize;

	/// STRUCTURING ELEMENTS
	size_t structuringElementStoreEndOffset = 0;
	const bool binaryCircle3x3[9] =
	{
		0, 1, 0,
		1, 1, 1,
		0, 1, 0
	};
	const size_t binaryCircle3x3Size = sizeof(binaryCircle3x3) / sizeof(binaryCircle3x3[0]);
	const size_t binaryCircle3x3Offset = structuringElementStoreEndOffset;
	cudaMemcpyToSymbol(structuringElementStore, binaryCircle3x3, binaryCircle3x3Size, binaryCircle3x3Offset);
	structuringElementStoreEndOffset += binaryCircle3x3Size;

	const bool binaryCircle5x5[25] =
	{
		0, 1, 1, 1, 0,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		0, 1, 1, 1, 0
	};
	const size_t binaryCircle5x5Size = sizeof(binaryCircle5x5) / sizeof(binaryCircle5x5[0]);
	const size_t binaryCircle5x5Offset = structuringElementStoreEndOffset;
	cudaMemcpyToSymbol(structuringElementStore, binaryCircle5x5, binaryCircle5x5Size, binaryCircle5x5Offset);
	structuringElementStoreEndOffset += binaryCircle5x5Size;

	const bool binaryCircle7x7[49] =
	{
		0, 0, 1, 1, 1, 0, 0,
		0, 1, 1, 1, 1, 1, 0,
		1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1,
		0, 1, 1, 1, 1, 1, 0,
		0, 0, 1, 1, 1, 0, 0,
	};
	const size_t binaryCircle7x7Size = sizeof(binaryCircle7x7) / sizeof(binaryCircle7x7[0]);
	const size_t binaryCircle7x7Offset = structuringElementStoreEndOffset;
	cudaMemcpyToSymbol(structuringElementStore, binaryCircle7x7, binaryCircle7x7Size, binaryCircle7x7Offset);
	structuringElementStoreEndOffset += binaryCircle7x7Size;

	activeCamera >> frame;
	unsigned char *greyscaleDataDevice2,
		*greyscaleDataDevice1,
		//*hsvImageDataDevice,
		*backgroundGreyscaleDataDevice,
		*backgroundGreyscaleBlurredDataDevice,
		*thresholdDataDevice,
		*erosionDataDevice,
		*dilationDataDevice,
		*ballTemplateDataDevice,
		//*ballTemplateBufferDataDevice,
		*bufferDataDevice,
		*displayDataDevice;
	cv::Mat greyscale1(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &greyscaleDataDevice1));
	cv::Mat greyscale2(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &greyscaleDataDevice2));
	//cv::Mat hsvImage(frame.size() * 3, CV_8U, createImageBuffer(frame.size().width * frame.size().height * 3, &hsvImageDataDevice));
	cv::Mat backgroundGreyscale(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &backgroundGreyscaleDataDevice));
	cv::Mat backgroundGreyscaleBlurred(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &backgroundGreyscaleBlurredDataDevice));
	cv::Mat thresholdImage(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &thresholdDataDevice));
	cv::Mat erosion(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &erosionDataDevice));
	cv::Mat dilation(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &dilationDataDevice));
	cv::Mat buffer(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &bufferDataDevice));
	cv::Mat display(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &displayDataDevice));

	int ballTemplateCropRadius = 40;
	cv::Size ballTemplateSize((ballTemplateCropRadius * 2) + 1, (ballTemplateCropRadius * 2) + 1);
	cv::Mat ballTemplate(ballTemplateSize, CV_8U, createImageBuffer(ballTemplateSize.width * ballTemplateSize.height, &ballTemplateDataDevice));
	//cv::Mat ballTemplateBuffer(ballTemplateSize, CV_8U, createImageBuffer(ballTemplateSize.width * ballTemplateSize.height, &ballTemplateBufferDataDevice));
	cudaMemcpyToSymbol(structuringElementStore, ballTemplateDataDevice, sizeof(ballTemplateDataDevice), binaryCircle7x7Offset + (sizeof(binaryCircle7x7) / sizeof(binaryCircle7x7[0])));
	const size_t ballTemplateOffset = binaryCircle7x7Offset + (sizeof(binaryCircle7x7) / sizeof(binaryCircle7x7[0]));

	cv::Mat greyscale, greyscalePrev;
	cv::Mat hsvImage;

	// object HSV thresholds
	cv::Vec3b lower_hsv = { 0, 0, 200 };
	cv::Vec3b upper_hsv = { 255, 255, 255 };

	int greyscaleState = 1;
	int keypressCur;

	cv::SimpleBlobDetector::Params params;
	// params.filterByArea = true;
	// params.minArea = 1;
	// params.maxArea = 300;
	params.filterByColor = true;
	params.blobColor = 255;
	cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
	std::vector<cv::KeyPoint> keypoints;


	while (1) {
		activeCamera >> frame;
		if (greyscaleState == 1) {
			cv::cvtColor(frame, greyscale1, CV_BGR2GRAY);
			greyscalePrev = greyscale2;
			greyscale = greyscale1;
			greyscaleState = 2;
		}
		else {
			cv::cvtColor(frame, greyscale2, CV_BGR2GRAY);
			greyscalePrev = greyscale1;
			greyscale = greyscale2;
			greyscaleState = 1;
		}

		// convert to HSV
		cv::cvtColor(frame, hsvImage, CV_BGR2HSV);
		cv::inRange(hsvImage, lower_hsv, upper_hsv, thresholdImage);

		dim3 cblocks(frame.size().width / 16, frame.size().height / 16);
		dim3 cthreads(16, 16);
		dim3 pblocks(frame.size().width * frame.size().height / 256);
		dim3 pthreads(256, 1);
		{
			switch (activeOperation) {
			case Normal:
				display = frame;
				break;
			case Greyscale:
				display = greyscale;
				break;
			case Subtraction:
				subtractImagesWrapper(cblocks, cthreads, greyscale.data, greyscalePrev.data, frame.size().width, frame.size().height, threshold, bufferDataDevice);
				display = buffer;
				break;
			case Background:
				if (backgroundSet) {
					// blur
					convolveWrapper(cblocks, cthreads, greyscale.data, frame.size().width, frame.size().height, 0, 0, gaussian5x5KernelOffset, 5, 5, bufferDataDevice);
					// background subtraction
					subtractImagesWrapper(cblocks, cthreads, bufferDataDevice, backgroundGreyscaleBlurred.data, frame.size().width, frame.size().height, threshold, thresholdDataDevice);
					// erode to remove noise
					erodeFilterWrapper(cblocks, cthreads, thresholdDataDevice, frame.size().width, frame.size().height, 0, 0, binaryCircle7x7Offset, 7, 7, erosionDataDevice);
					// dilate
					dilateFilterWrapper(cblocks, cthreads, erosionDataDevice, frame.size().width, frame.size().height, 0, 0, binaryCircle7x7Offset, 7, 7, dilationDataDevice);

					// blob detector
					detector->detect(dilation, keypoints);
					// ignore if multiple blobs found
					if (keypoints.size() > 1) {
						std::cout << "more than one keypoint found" << std::endl;
					}
					else if (keypoints.size() == 1) {
						std::cout << "just one keypoint found - " << keypoints[0].pt << std::endl;
						// grab the center
						cv::Point centerPoint = keypoints[0].pt;
						// pick bounds for blob template
						int left = floor(centerPoint.x) - ballTemplateCropRadius;
						if (left < 0) left = 0;
						int right = floor(centerPoint.x) + ballTemplateCropRadius;
						if (right > frame.size().width - 1) right = frame.size().width - 1;
						int top = floor(centerPoint.y) - ballTemplateCropRadius;
						if (top < 0) top = 0;
						int bottom = floor(centerPoint.y) + ballTemplateCropRadius;
						if (bottom > frame.size().height - 1) bottom = frame.size().height - 1;
						// build the template
						cv::Mat part(
							dilation,
							cv::Range(top, bottom),
							cv::Range(left, right));
						ballTemplate = part;

						cudaMemcpyToSymbol(structuringElementStore, ballTemplateDataDevice, sizeof(ballTemplateDataDevice), ballTemplateOffset);

						char h_range = 15;
						char s_range = 100;
						char v_range = 100;
						cv::Vec3b colorOfBall = hsvImage.at<cv::Vec3b>(centerPoint);
						lower_hsv = cv::Vec3b(max(colorOfBall[0] - h_range, 0), max(colorOfBall[1] - s_range, 0), max(colorOfBall[2] - v_range, 0));
						upper_hsv = cv::Vec3b(min(colorOfBall[0] + h_range, 255), min(colorOfBall[1] + s_range, 255), min(colorOfBall[2] + v_range, 255));


						cv::imshow("Template Buffer", ballTemplate);
						cv::circle(buffer, keypoints[0].pt, 40, cv::Scalar(255, 0, 0), 2);
					}
					else {
						std::cout << "nothing found - " << std::endl;
					}

					display = dilation;
				}
				else {
					std::cout << "Take Background Picture by pressing 'p' " << std::endl;
					while (1) if (cv::waitKey(1) == 112) break;
					activeCamera >> background;
					cv::cvtColor(background, backgroundGreyscale, CV_BGR2GRAY);
					convolveWrapper(cblocks, cthreads, backgroundGreyscale.data, frame.size().width, frame.size().height, 0, 0, gaussian5x5KernelOffset, 5, 5, backgroundGreyscaleBlurredDataDevice);
					backgroundSet = 1;
				}
				break;
			case Tracking:
				erodeFilterWrapper(cblocks, cthreads, thresholdDataDevice, frame.size().width, frame.size().height, 0, 0, binaryCircle5x5Offset, 5, 5, erosionDataDevice);
				dilateFilterWrapper(cblocks, cthreads, erosionDataDevice, frame.size().width, frame.size().height, 0, 0, binaryCircle5x5Offset, 5, 5, dilationDataDevice);
				dilateFilterWrapper(cblocks, cthreads, dilationDataDevice, frame.size().width, frame.size().height, 0, 0, binaryCircle5x5Offset, 5, 5, bufferDataDevice);
				//invertBIWrapper(cblocks, cthreads, bufferDataDevice, frame.size().width, frame.size().height, dilationDataDevice);
				//logicalAndWrapper(cblocks, cthreads, dilationDataDevice, bufferDataDevice, erosionDataDevice, frame.size().width, frame.size().height);
				//erodeFilterWrapper(cblocks, cthreads, bufferDataDevice, frame.size().width, frame.size().height, 0, 0, ballTemplateOffset, ballTemplate.size().width, ballTemplate.size().height, erosionDataDevice);
				display = buffer;
				break;
			default:
				break;
			}
		}

		if (recording)
			oVideoWriter.write(display);


		if (display.size().height > 0) {
			cv::namedWindow("Convolution", cv::WINDOW_AUTOSIZE);
			cv::imshow("Convolution", display);
		}

		keypressCur = cv::waitKey(1);
		if (keypressCur < 255) {
			keypress = keypressCur;
			handleKeypress(frame);
		}

		if (keypress == 27) break;
	}
	cudaFreeHost(greyscale1.data);
	cudaFreeHost(greyscale2.data);
	cudaFreeHost(backgroundGreyscale.data);
	cudaFreeHost(backgroundGreyscaleBlurred.data);
	cudaFreeHost(thresholdImage.data);
	cudaFreeHost(erosion.data);
	cudaFreeHost(dilation.data);
	cudaFreeHost(ballTemplate.data);
	//cudaFreeHost(ballTemplateBuffer.data);
	cudaFreeHost(buffer.data);
	cudaFreeHost(display.data);

	oVideoWriter.release();

	return 0;
}

unsigned char *createImageBuffer(unsigned int bytes, unsigned char **devicePtr) {

	unsigned char *ptr = NULL;
	cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaHostAlloc(&ptr, bytes, cudaHostAllocMapped);
	cudaHostGetDevicePointer(devicePtr, ptr, 0);
	return ptr;
}


void invertBIWrapper(dim3 blocks, dim3 threads, unsigned char *source, int width, int height, unsigned char *destination)
{
#if TIME_GPU
	cudaEventRecord(start);
#endif

	invertBI << <blocks, threads >> >(source, width, height, destination);
	cudaDeviceSynchronize();

#if TIME_GPU
	cudaEventRecord(stop);
	float ms = 0.0f;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	std::cout << "Elapsed GPU time: " << ms << " milliseconds" << std::endl;
#endif
}
__global__ void invertBI(unsigned char *source, int width, int height, unsigned char *destination)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int i = (y * width) + x;

	destination[i] = (unsigned char) ( source[i] > 0 ) ? 0 : 255;
}


void logicalAndWrapper(dim3 blocks, dim3 threads, unsigned char *a, unsigned char *b, unsigned char *c, int width, int height) {
#if TIME_GPU
	cudaEventRecord(start);
#endif

	logicalAnd << <blocks, threads >> >(a, b, c, width, height);
	cudaDeviceSynchronize();

#if TIME_GPU
	cudaEventRecord(stop);
	float ms = 0.0f;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	std::cout << "Elapsed GPU time: " << ms << " milliseconds" << std::endl;
#endif
}
__global__ void logicalAnd(unsigned char *a, unsigned char *b, unsigned char *c, int width, int height)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int i = (y * width) + x;

	c[i] = (unsigned char) ((a[i]>0) && (b[i]>0) ? 255 : 0);
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


void convolveWrapper(dim3 blocks, dim3 threads, unsigned char *source, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *destination) {
#if TIME_GPU
	cudaEventRecord(start);
#endif

	convolve << <blocks, threads >> > (source, width, height, paddingX, paddingY, kOffset, kWidth, kHeight, destination);
	cudaDeviceSynchronize();

#if TIME_GPU
	cudaEventRecord(stop);
	float ms = 0.0f;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	std::cout << "Elapsed GPU time: " << ms << " milliseconds" << std::endl;
#endif
}
__global__ void convolve(unsigned char *source, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *destination) {

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


				sum += w * float(source[((y + j) * width) + (x + i)]);
			}
		}
	}

	destination[(y * width) + x] = (unsigned char)sum;
}


void subtractImagesWrapper(dim3 blocks, dim3 threads, unsigned char *img1, unsigned char *img2, int width, int height, float threshold, unsigned char *destination) {
#if TIME_GPU
	cudaEventRecord(start);
#endif

	subtractImages << <blocks, threads >> > (img1, img2, width, height, threshold, destination);
	cudaDeviceSynchronize();

#if TIME_GPU
	cudaEventRecord(stop);
	float ms = 0.0f;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	std::cout << "Elapsed GPU time: " << ms << " milliseconds" << std::endl;
#endif
}
__global__ void subtractImages(unsigned char *img1, unsigned char *img2, int width, int height, float threshold, unsigned char *destination) {

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	float pixelDifference = abs(img1[(y * width) + x] - img2[(y * width) + x]);
	if (pixelDifference > threshold) {
		destination[(y * width) + x] = 255.0f;
	}
	else {
		destination[(y * width) + x] = 0.0f;
	}
}


void erodeFilterWrapper(dim3 blocks, dim3 threads, unsigned char *source, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *destination) {
#if TIME_GPU
	cudaEventRecord(start);
#endif

	erodeFilter << <blocks, threads >> > (source, width, height, paddingX, paddingY, kOffset, kWidth, kHeight, destination);
	cudaDeviceSynchronize();

#if TIME_GPU
	cudaEventRecord(stop);
	float ms = 0.0f;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	std::cout << "Elapsed GPU time: " << ms << " milliseconds" << std::endl;
#endif
}
__global__ void erodeFilter(unsigned char *source, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *destination)
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
				bool w = (structuringElementStore[(kj * kWidth) + ki + kOffset] > 0);
				if (w)
				{
					erode = !(source[((y + j) * width) + (x + i)] > 0);
				}
			}
		}
	}

	destination[(y * width) + x] = (erode) ? 0 : 255;
}

void dilateFilterWrapper(dim3 blocks, dim3 threads, unsigned char *source, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *destination) {
#if TIME_GPU
	cudaEventRecord(start);
#endif
	memset(destination, 0, width*height);
	dilateFilter << <blocks, threads >> > (source, width, height, paddingX, paddingY, kOffset, kWidth, kHeight, destination);
	cudaDeviceSynchronize();

#if TIME_GPU
	cudaEventRecord(stop);
	float ms = 0.0f;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	std::cout << "Elapsed GPU time: " << ms << " milliseconds" << std::endl;
#endif
}
__global__ void dilateFilter(unsigned char *source, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *destination)
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
		if (source[(y * width) + x] > 0)
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
						destination[((y + j) * width) + (x + i)] = 255;
					}
				}
			}
		}
	}
}

void centerOfMass(cv::Point centerPoint, unsigned char *source, int width, int height, int outlierDist, int maxPoints)
{
	// variables
	std::vector<cv::Point> points; // points that are on
	float xi, yi, xf, yf; // initial and final center of mass coordinates
	int pointsInRange; // points contained within the outlier region

	// build point arrays
	for(int y = 0, pIndex = 0; y < height && pIndex < maxPoints; y++)
	{
		for(int x = 0; x < width && pIndex < maxPoints; x++)
		{
			if(source[(y*width) + x] > 0)
			{
				cv::Point myPoint(x, y);
				points.push_back( myPoint );
			}
		}
	}

	if( points.size() > 0)
	{
		// find initial center of mass
		for(int i = 0; i < points.size(); i++)
		{
			cv::Point p = points[i];
			xi += p.x;
			yi += p.y;
		}
		xi /= points.size();
		yi /= points.size();

		// find new center of mass with outlier
		pointsInRange = 0;
		for(int i = 0; i < points.size(); i++)
		{
			cv::Point p = points[i];
			if( abs( p.x - xi ) < outlierDist && abs( p.y - yi ) < outlierDist )
			{
				xf += p.x;
				yf += p.y;
				pointsInRange++;
			}
		}
		xf /= pointsInRange;
		yf /= pointsInRange;

		centerPoint.x = xf;
		centerPoint.y = yf;
	}

	return;
}

void handleKeypress(cv::Mat frame) {
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

void convolveWrapper(dim3 blocks, dim3 threads, unsigned char *source, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *destination);
__global__ void convolve(unsigned char *source, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *destination);

void pythagoras_slow(unsigned char *a, unsigned char *b, unsigned char *c, int width, int height);
void convolve_slow(unsigned char *source, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, const float *kernel, unsigned char *destination);

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
cv::VideoCapture activeCamera = camera_front;

int activeProcessing = 0; /* 0 = Use the GPU. 1 = Use the CPU */

int main() {

	cv::Mat frame;
	if ((!camera_front.isOpened()) || (!camera_back.isOpened()))
		return -1;

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
					convolveWrapper(cblocks, cthreads, greyscaleDataDevice, frame.size().width, frame.size().height, 0, 0, gaussian5x5KernelOffset, 5, 5, blurredDataDevice);
					display = blurred;
					break;
				case Embossed:
					convolveWrapper(cblocks, cthreads, greyscaleDataDevice, frame.size().width, frame.size().height, 0, 0, emboss3x3KernelOffset, 3, 3, embossedDataDevice);
					display = embossed;
					break;
				case Outline:
					convolveWrapper(cblocks, cthreads, greyscaleDataDevice, frame.size().width, frame.size().height, 0, 0, outline3x3KernelOffset, 3, 3, outlineDataDevice);
					display = outline;
					break;
				case Sobel:
					convolveWrapper(cblocks, cthreads, greyscaleDataDevice, frame.size().width, frame.size().height, 0, 0, gaussian5x5KernelOffset, 5, 5, blurredDataDevice);
					convolveWrapper(cblocks, cthreads, blurredDataDevice, frame.size().width, frame.size().height, 0, 0, leftSobel3x3KernelOffset, 3, 3, leftSobelDataDevice);
					convolveWrapper(cblocks, cthreads, blurredDataDevice, frame.size().width, frame.size().height, 0, 0, topSobel3x3KernelOffset, 3, 3, topSobelDataDevice);
					pythagorasWrapper(pblocks, pthreads, leftSobelDataDevice, topSobelDataDevice, sobelDataDevice);
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
				convolve_slow(greyscaleDataDevice, frame.size().width, frame.size().height, 0, 0, gaussian5x5KernelOffset, 5, 5, gaussian5x5Kernel, blurredDataDevice);
				display = blurred;
				break;
			case Embossed:
				convolve_slow(greyscaleDataDevice, frame.size().width, frame.size().height, 0, 0, emboss3x3KernelOffset, 3, 3, emboss3x3Kernel, embossedDataDevice);
				display = embossed;
				break;
			case Outline:
				convolve_slow(greyscaleDataDevice, frame.size().width, frame.size().height, 0, 0, outline3x3KernelOffset, 3, 3, outline3x3Kernel, outlineDataDevice);
				display = outline;
				break;
			case Sobel:
				convolve_slow(greyscaleDataDevice, frame.size().width, frame.size().height, 0, 0, gaussian5x5KernelOffset, 5, 5, gaussian5x5Kernel, blurredDataDevice);
				convolve_slow(blurredDataDevice, frame.size().width, frame.size().height, 0, 0, leftSobel3x3KernelOffset, 3, 3, leftSobel3x3Kernel, leftSobelDataDevice);
				convolve_slow(blurredDataDevice, frame.size().width, frame.size().height, 0, 0, topSobel3x3KernelOffset, 3, 3, topSobel3x3Kernel, topSobelDataDevice);
				pythagoras_slow(leftSobelDataDevice, topSobelDataDevice, sobelDataDevice, frame.size().width, frame.size().height);
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


void convolveWrapper(dim3 blocks, dim3 threads, unsigned char *source, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *destination) {
#if TIME_GPU
	cudaEventRecord(start);
#endif

	convolve << <blocks, threads >> > (source, width, height, paddingX, paddingY, kOffset, kWidth, kHeight, destination);
	cudaDeviceSynchronize();

#if TIME_GPU
	cudaEventRecord(stop);
	float ms = 0.0f;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	std::cout << "Elapsed GPU time: " << ms << " milliseconds" << std::endl;
#endif
}
__global__ void convolve(unsigned char *source, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *destination) {

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


				sum += w * float(source[((y + j) * width) + (x + i)]);
			}
		}
	}

	destination[(y * width) + x] = (unsigned char)sum;
}


void pythagoras_slow(unsigned char *a, unsigned char *b, unsigned char *c, int width, int height)
{
	for (int i = 0; i < width * height; i++) {

		float af = float(a[i]);
		float bf = float(b[i]);
		c[i] = (unsigned char)sqrtf(af*af + bf*bf);
	}

}

void convolve_slow(unsigned char *source, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, const float *kernel, unsigned char *destination) {

	/*for (int y = 0; y < height; y++)
	{
	for (int x = 0; x < width; x++)
	{
	source[(y * width) + x]
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

					sum += w * source[((y + j) * width) + (x + i)];

					/*if ((k > 2000) && (k < 2010)) {
					std::cout << "Kernel Index " << (int)(kj * kWidth) + ki << "   val at Kernel Index " << w <<  "    K " << k <<  "   Source Index " << ((y + j) * width) + (x + i) << "    Sum " << sum << std::endl;
					}*/
				}
			}
		}

		sum = (sum < 0) ? 0 : sum;
		sum = (sum > 255) ? 255 : sum;
		destination[k] = sum;
		/*if ((k > 2000) && (k < 2010)) {
		std::cout << (int) source[k] << "    " << (int) destination[k] << "    " << std::endl;
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

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

__constant__ float convolutionKernelStore[256];

void captureBackgroundImage();
unsigned char *createImageBuffer(unsigned int bytes, unsigned char **devicePtr);
__global__ void pythagoras(unsigned char *a, unsigned char *b, unsigned char *c);
__global__ void convolve(unsigned char *source, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *destination);
__global__ void subtractImages(unsigned char *img1, unsigned char *img2, int width, int height, float threshold, unsigned char *destination);

void handleKeypress();
int keypress, backgroundSet = 0;
cv::Mat background, backgroundGreyscale;


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

int activeProcessing = 0; /* 0 = Use the GPU. 1 = Use the CPU */
float threshold = 100;

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


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	const float gaussianKernel[25] =
	{
		2.f / 159.f,  4.f / 159.f,  5.f / 159.f,  4.f / 159.f, 2.f / 159.f,
		4.f / 159.f,  9.f / 159.f, 12.f / 159.f,  9.f / 159.f, 4.f / 159.f,
		5.f / 159.f, 12.f / 159.f, 15.f / 159.f, 12.f / 159.f, 5.f / 159.f,
		4.f / 159.f,  9.f / 159.f, 12.f / 159.f,  9.f / 159.f, 4.f / 159.f,
		2.f / 159.f,  4.f / 159.f,  5.f / 159.f,  4.f / 159.f, 2.f / 159.f,
	};
	cudaMemcpyToSymbol(convolutionKernelStore, gaussianKernel, sizeof(gaussianKernel), 0);
	const size_t gaussianKernelOffset = 0;

	const float embossKernel[9] =
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
	cudaMemcpyToSymbol(convolutionKernelStore, embossKernel, sizeof(embossKernel), sizeof(gaussianKernel));
	const size_t embossKernelOffset = sizeof(gaussianKernel) / sizeof(float);

	const float outlineKernel[9] =
	{
		-1.f, -1.f, -1.f,
		-1.f, 8.f, -1.f,
		-1.f, -1.f, -1.f,
	};
	cudaMemcpyToSymbol(convolutionKernelStore, outlineKernel, sizeof(outlineKernel), sizeof(gaussianKernel) + sizeof(embossKernel));
	const size_t outlineKernelOffset = sizeof(embossKernel) / sizeof(float) + embossKernelOffset;

	const float leftSobelKernel[9] =
	{
		1.f, 0.f, -1.f,
		2.f, 0.f, -2.f,
		1.f, 0.f, -1.f,
	};
	cudaMemcpyToSymbol(convolutionKernelStore, leftSobelKernel, sizeof(leftSobelKernel), sizeof(gaussianKernel) + sizeof(embossKernel) + sizeof(outlineKernel));
	const size_t leftSobelKernelOffset = sizeof(outlineKernel) / sizeof(float) + outlineKernelOffset;
	const float topSobelKernel[9] =
	{
		1.f, 2.f, 1.f,
		0.f, 0.f, 0.f,
		-1.f, -2.f, -1.f,
	};
	cudaMemcpyToSymbol(convolutionKernelStore, topSobelKernel, sizeof(topSobelKernel), sizeof(gaussianKernel) + sizeof(embossKernel) + sizeof(outlineKernel) + sizeof(leftSobelKernel));
	const size_t topSobelKernelOffset = sizeof(leftSobelKernel) / sizeof(float) + leftSobelKernelOffset;

	activeCamera >> frame;
	unsigned char *greyscaleDataDevice2, *greyscaleDataDevice1, *backgroundGreyscaleDataDevice, *bufferDataDevice, *displayDataDevice;
	cv::Mat greyscale1(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &greyscaleDataDevice1));
	cv::Mat greyscale2(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &greyscaleDataDevice2));
	cv::Mat backgroundGreyscale(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &backgroundGreyscaleDataDevice));
	cv::Mat buffer(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &bufferDataDevice));
	cv::Mat display(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &displayDataDevice));
	cv::Mat greyscale, greyscalePrev;

	int greyscaleState = 1;
	int keypressCur;

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
		


		if (activeProcessing == 0) {
			dim3 cblocks(frame.size().width / 16, frame.size().height / 16);
			dim3 cthreads(16, 16);
			dim3 pblocks(frame.size().width * frame.size().height / 256);
			dim3 pthreads(256, 1);
			cudaEventRecord(start);
			{
				switch (activeOperation) {
				case Normal:
					display = frame;
					break;
				case Greyscale:
					display = greyscale;
					break;
				case Subtraction:	
					subtractImages << <cblocks, cthreads >> > (greyscale.data, greyscalePrev.data, frame.size().width, frame.size().height, threshold, bufferDataDevice);
					display = buffer;
					break;
				case Background:
					if (backgroundSet) {
						subtractImages << <cblocks, cthreads >> > (backgroundGreyscale.data, greyscale.data, frame.size().width, frame.size().height, threshold, bufferDataDevice);
						display = buffer;
					}
					else std::cout << "Please use 'p' to capture background image" << std::endl;
					break;
				case Tracking:
					
					break;
				default:
					break;
				}
				cudaDeviceSynchronize();
			}
			cudaEventRecord(stop);
			float ms = 0.0f;
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&ms, start, stop);
			std::cout << "Elapsed GPU time: " << ms << " milliseconds" << std::endl;
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

	cudaFreeHost(greyscale1.data);
	cudaFreeHost(greyscale2.data);
	cudaFreeHost(backgroundGreyscale.data);
	cudaFreeHost(buffer.data);
	cudaFreeHost(display.data);

	return 0;
}

void captureBackgroundImage() {
	std::cout << "Take Background Picture by pressing 'p' " << std::endl;
	while (1) if (cv::waitKey(1) == 112) break;
	activeCamera >> background;
	cv::cvtColor(background, backgroundGreyscale, CV_BGR2GRAY);
	backgroundSet = 1;
}

unsigned char *createImageBuffer(unsigned int bytes, unsigned char **devicePtr) {

	unsigned char *ptr = NULL;
	cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaHostAlloc(&ptr, bytes, cudaHostAllocMapped);
	cudaHostGetDevicePointer(devicePtr, ptr, 0);
	return ptr;
}

__global__ void pythagoras(unsigned char *a, unsigned char *b, unsigned char *c)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	float af = float(a[idx]);
	float bf = float(b[idx]);

	c[idx] = (unsigned char)sqrtf(af*af + bf*bf);
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

__global__ void subtractImages(unsigned char *img1, unsigned char *img2, int width, int height, float threshold, unsigned char *destination) {

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	float pixel = abs(img1[(y * width) + x] - img2[(y * width) + x]);
	if (pixel > threshold) {
		destination[(y * width) + x] = 255.0f;
	}
	else {
		destination[(y * width) + x] = 0.0f;
	}
}

void handleKeypress() {

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

	case 116: /* t */
		threshold += 5;
		break;
	case 121: /* y */
		threshold -= 5;
		break;

	case 112: /* p */
		captureBackgroundImage();
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

__constant__ float convolutionKernelStore[256];

unsigned char *createImageBuffer(unsigned int bytes, unsigned char **devicePtr);
__global__ void pythagoras(unsigned char *a, unsigned char *b, unsigned char *c);
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

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	clock_t start_clock, end_clock;
	double cpu_time_used;

	const float gaussianKernel[25] =
	{
		2.f / 159.f,  4.f / 159.f,  5.f / 159.f,  4.f / 159.f, 2.f / 159.f,
		4.f / 159.f,  9.f / 159.f, 12.f / 159.f,  9.f / 159.f, 4.f / 159.f,
		5.f / 159.f, 12.f / 159.f, 15.f / 159.f, 12.f / 159.f, 5.f / 159.f,
		4.f / 159.f,  9.f / 159.f, 12.f / 159.f,  9.f / 159.f, 4.f / 159.f,
		2.f / 159.f,  4.f / 159.f,  5.f / 159.f,  4.f / 159.f, 2.f / 159.f,
	};
	cudaMemcpyToSymbol(convolutionKernelStore, gaussianKernel, sizeof(gaussianKernel), 0);
	const size_t gaussianKernelOffset = 0;

	const float embossKernel[9] =
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
	cudaMemcpyToSymbol(convolutionKernelStore, embossKernel, sizeof(embossKernel), sizeof(gaussianKernel));
	const size_t embossKernelOffset = sizeof(gaussianKernel) / sizeof(float);

	const float outlineKernel[9] =
	{
		-1.f, -1.f, -1.f,
		-1.f, 8.f, -1.f,
		-1.f, -1.f, -1.f,
	};
	cudaMemcpyToSymbol(convolutionKernelStore, outlineKernel, sizeof(outlineKernel), sizeof(gaussianKernel) + sizeof(embossKernel));
	const size_t outlineKernelOffset = sizeof(embossKernel) / sizeof(float) + embossKernelOffset;

	const float leftSobelKernel[9] =
	{
		1.f, 0.f, -1.f,
		2.f, 0.f, -2.f,
		1.f, 0.f, -1.f,
	};
	cudaMemcpyToSymbol(convolutionKernelStore, leftSobelKernel, sizeof(leftSobelKernel), sizeof(gaussianKernel) + sizeof(embossKernel) + sizeof(outlineKernel));
	const size_t leftSobelKernelOffset = sizeof(outlineKernel) / sizeof(float) + outlineKernelOffset;
	const float topSobelKernel[9] =
	{
		1.f, 2.f, 1.f,
		0.f, 0.f, 0.f,
		-1.f, -2.f, -1.f,
	};
	cudaMemcpyToSymbol(convolutionKernelStore, topSobelKernel, sizeof(topSobelKernel), sizeof(gaussianKernel) + sizeof(embossKernel) + sizeof(outlineKernel) + sizeof(leftSobelKernel));
	const size_t topSobelKernelOffset = sizeof(leftSobelKernel) / sizeof(float) + leftSobelKernelOffset;

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
			cudaEventRecord(start);
			{
				switch (activeKernel) {
				case Normal:
					display = frame;
					break;
				case Greyscale:
					display = greyscale;
					break;
				case Blurred:
					convolve << <cblocks, cthreads >> > (greyscaleDataDevice, frame.size().width, frame.size().height, 0, 0, gaussianKernelOffset, 5, 5, blurredDataDevice);
					display = blurred;
					break;
				case Embossed:
					convolve << <cblocks, cthreads >> > (greyscaleDataDevice, frame.size().width, frame.size().height, 0, 0, embossKernelOffset, 3, 3, embossedDataDevice);
					display = embossed;
					break;
				case Outline:
					convolve << <cblocks, cthreads >> > (greyscaleDataDevice, frame.size().width, frame.size().height, 0, 0, outlineKernelOffset, 3, 3, outlineDataDevice);
					display = outline;
					break;
				case Sobel:
					convolve << <cblocks, cthreads >> > (greyscaleDataDevice, frame.size().width, frame.size().height, 0, 0, gaussianKernelOffset, 5, 5, blurredDataDevice);
					convolve << <cblocks, cthreads >> > (blurredDataDevice, frame.size().width, frame.size().height, 0, 0, leftSobelKernelOffset, 3, 3, leftSobelDataDevice);
					convolve << <cblocks, cthreads >> > (blurredDataDevice, frame.size().width, frame.size().height, 0, 0, topSobelKernelOffset, 3, 3, topSobelDataDevice);
					pythagoras << <pblocks, pthreads >> >(leftSobelDataDevice, topSobelDataDevice, sobelDataDevice);
					display = sobel;
					break;
				default:
					break;
				}
				cudaDeviceSynchronize();
			}
			cudaEventRecord(stop);
			float ms = 0.0f;
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&ms, start, stop);
			std::cout << "Elapsed GPU time: " << ms << " milliseconds" << std::endl;
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
				convolve_slow(greyscaleDataDevice, frame.size().width, frame.size().height, 0, 0, gaussianKernelOffset, 5, 5, gaussianKernel, blurredDataDevice);
				display = blurred;
				break;
			case Embossed:
				convolve_slow(greyscaleDataDevice, frame.size().width, frame.size().height, 0, 0, embossKernelOffset, 3, 3, embossKernel, embossedDataDevice);
				display = embossed;
				break;
			case Outline:
				convolve_slow(greyscaleDataDevice, frame.size().width, frame.size().height, 0, 0, outlineKernelOffset, 3, 3, outlineKernel, outlineDataDevice);
				display = outline;
				break;
			case Sobel:
				convolve_slow(greyscaleDataDevice, frame.size().width, frame.size().height, 0, 0, gaussianKernelOffset, 5, 5, gaussianKernel, blurredDataDevice);
				convolve_slow(blurredDataDevice, frame.size().width, frame.size().height, 0, 0, leftSobelKernelOffset, 3, 3, leftSobelKernel, leftSobelDataDevice);
				convolve_slow(blurredDataDevice, frame.size().width, frame.size().height, 0, 0, topSobelKernelOffset, 3, 3, topSobelKernel, topSobelDataDevice);
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

__global__ void pythagoras(unsigned char *a, unsigned char *b, unsigned char *c)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	float af = float(a[idx]);
	float bf = float(b[idx]);

	c[idx] = (unsigned char)sqrtf(af*af + bf*bf);
}

void pythagoras_slow(unsigned char *a, unsigned char *b, unsigned char *c, int width, int height)
{
	for (int i = 0; i < width * height; i++) {

		float af = float(a[i]);
		float bf = float(b[i]);
		c[i] = (unsigned char)sqrtf(af*af + bf*bf);
	}

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
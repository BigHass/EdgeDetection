# Edge Detection

* Edge detction  is a common image processing technique that is used to identify the edges in an image. An edge is a boundary between two distinct regions in an image.   It is often used to highlight features in an image or to separate objects from their background.

* There are several algorithms that can be used for edge detection, including the Sobel operator, Canny edge detector, and Prewitt operator. These algorithms work by     analyzing the intensity values of the pixels in the image and identifying sudden changes in intensity as edges.

* Edge detection can be useful for a variety of applications, including *image segmentation*, *object recognition*, and *image enhancement*.

* In this image, we show the effect of applied edge detection:
      <p align="center">
            <img src="/assets/images/edgeDetection.png" alt="edgeDetection image">
      </p>


# Convolution Process

* In image processing, convolution is a mathematical operation that is used to modify the intensity values of pixels in an image. It involves multiplying 
the intensity values of the pixels in the image by a set of coefficients, or "weights", and summing the results to produce a new intensity value for each pixel.

* Convolution is often used to apply various types of filters to an image, such as edge detection filters, blur filters, and sharpening filters. It is also used in
image compression, image recognition, and other applications.

* To perform convolution on an image, the following steps are typically followed:

  1. Define a convolution kernel, which is a small matrix of weights that will be used to modify the intensity values of the pixels.
  2. Slide the kernel over the image, one pixel at a time, and multiply the intensity values of the overlapping pixels by the corresponding weights in the kernel.
  3. Sum the results of the multiplications to produce a new intensity value for the center pixel of the kernel.
  4. Repeat the process for all pixels in the image.
  5. The resulting image will have the same size as the original image, but the intensity values of the pixels will have been modified according to the weights in the        convolution kernel.
  
  * We can represent the above steps mathematically with the following equation:
  
    **Output[i, j] = sum(Input[x, y] * Kernel[x - i, y - j])**
    
    Where:
      - Output[i, j] is the intensity value of the pixel at position (i, j) in the output image.
      - Input[x, y] is the intensity value of the pixel at position (x, y) in the input image.
      - Kernel[x - i, y - j] is the weight of the kernel at position (x - i, y - j) relative to the center of the kernel.
      - The sum is taken over all x and y within the bounds of the kernel.
      - This equation can be used to calculate the intensity value of each pixel in the output image by sliding the kernel over the input image, multiplying the                 overlapping pixels by the corresponding weights in the kernel, and summing the results.

* We are using this image to demonstrate the convolution processing:
              
     <p align="center">
        <img src="/assets/images/2D_Convolution_Animation.gif" alt="convolution image">
     </p>
    
# MPI and Pthread 

* This code is using the Message Passing Interface (MPI) library to perform parallel processing of an image using the Sobel edge detection algorithm. The input image is read in by the master process, then broadcast to all other processes. Each process receives a portion of the image and applies the Sobel algorithm on it in a separate thread. The resulting image slices are then gathered by the master process and combined to form the final output image, which is then written to a file. The stb_image_write library is used to write the output image in the Portable Network Graphics (PNG) format.
  
# Requirements 
      * A C++ compiler
      * The MPI library
      * The stb_image_write library
# Cmppiling and Running 
* To compile the code, use the following command:
   - mpic++ -std=c++17 -pthread main.cpp -o sobel


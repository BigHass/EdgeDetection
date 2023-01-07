#pragma once

#include <cmath>
#include <future>
#include "threadArgs.h"

namespace h2o {
    // Combines the two input images using the hypotenuse formula.
    // The resulting image is the same size as the input images.
    Image::Ptr combine(
            const Image::Ptr &gx, // The first input image
            const Image::Ptr &gy, // The second input image.
            int imgRows, // The number of rows in the input images.
            int imgColumns // The number of columns in the input images.
    ) {
        // Creates an empty output image.
        auto image = Image::empty(imgRows, imgColumns);
        for (int i = 0; i < imgRows * imgColumns; i++) {
            // Calculates the final pixel value using the hypotenuse formula.
            double finalValue =
                    hypot(
                            static_cast<unsigned char>(gx->at(i)),
                            static_cast<unsigned char>(gy->at(i))
                    );
            // Clamps the final pixel value to the range [0, 255].
            if (finalValue > 255) {
                finalValue = 255;
            }
            // Sets the pixel value in the output image.
            image->at(i) = static_cast<std::byte>(finalValue);
        }
        // Returns the output image.
        return image;
    }

    std::byte clamp(float pixel) {
        // Clamps the pixel value to the range [0, 255].
        if (pixel < 0) {
            pixel = 0;
        } else if (pixel > 255) {
            pixel = 255;
        }

        return static_cast<std::byte>(pixel);
    }
    // Applies a convolution operation to a portion of the input image using the given kernel.
    Image::Ptr convolve(
            const Image::Ptr &image,
            int rowStart, // The first row of the desired portion of the input image.
            int rowEnd, // The last row of the desired portion of the input image. 
            std::span<const int> kernel // The convolution kernel.
    ) {
        // The indices of the center of the kernel.
        constexpr int kRowCenter = 1;
        constexpr int kColumnCenter = 1;
        // The size of the kernel.
        constexpr int kSize = 3;
        // The number of columns in the input image.
        const auto columns = image->columns();
        // Creates an empty output image with the desired number of rows and the same number of columns as the input image.
        auto partialConvolvedImage = Image::empty((rowEnd - rowStart), columns);
        // Iterates through the desired rows and columns of the input image.
        for (int i = rowStart; i < rowEnd; i++) {
            for (int j = 0; j < columns; j++) {
                float newPixel = 0;
                for (int ki = 0; ki < kSize; ki++) {
                    for (int kj = 0; kj < kSize; kj++) {
                        // Centers the current pixel over the kernel.
                        int iCentered = i - kRowCenter + ki;
                        int jCentered = j - kColumnCenter + kj;

                        // Pads with zero if the kernel extends beyond the bounds of the input image.
                        if ((iCentered < 0 || iCentered >= image->rows()) || (jCentered < 0 || jCentered >= columns)) {
                            continue;
                        }
                        // Accumulates the new pixel value.
                        newPixel += static_cast<int>(image->at(iCentered * columns + jCentered)) *
                                    static_cast<int>(kernel[ki * kSize + kj]);
                    }
                }
                // Clamps the new pixel value to the range [0, 255] and sets it in the output image.
                partialConvolvedImage->at((i - rowStart) * columns + j) = clamp(newPixel);
            }
        }

        return partialConvolvedImage;
    }
    // Performs the Sobel edge detection algorithm on a portion of the input image using separate threads for the x and y convolutions.
    void sobel(const ThreadArgs &task, std::promise<Image::Ptr> &&image_promise) {
        // The first and last rows of the desired portion of the input image.
        int rowStart = task.rowStart;
        int rowEnd = task.rowEnd;
        // The convolution kernels for the x and y directions.
        constexpr std::array<int, 9> sobelX{-1, 0, 1, -2, 0, 2, -1, 0, 1};
        constexpr std::array<int, 9> sobelY{1, 2, 1, 0, 0, 0, -1, -2, -1};
        // Performs a convolution on the input image in the x direction.
        auto gx = convolve(task.image, rowStart, rowEnd, sobelX);
        // Performs a convolution on the input image in the y direction.
        auto gy = convolve(task.image, rowStart, rowEnd, sobelY);
        // Throws an error if the x and y convolved images have different sizes.
        if (gx->rows() != gy->rows() || gy->columns() != gy->columns())
            throw std::runtime_error("Image parts do not have the same size");
        // Combines the x and y convolved images using the combine function.
        image_promise.set_value(combine(gx, gy, rowEnd - rowStart, gx->columns()));
    }


}


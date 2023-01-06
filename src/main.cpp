#include <thread>

#include "stb_image_write.h"

#include "image.h"
#include "sobel.h"
#include "threadArgs.h"
#include "mpi_handler.h"

int main(int argc, char *argv[]) {

    h2o::MPIHandler handler(argc, argv);
    // Declares pointers to Image objects for the input and output images.
    h2o::Image::Ptr image1D;
    h2o::Image::Ptr convolvedImage;

    int MASTER = handler.numOfProcs - 1;
    // Declares variables for storing the number of rows and columns in the input image.
    int imgColumns{}, imgRows{};
    int desiredChannels = 1;

    // Declares vectors for use in the MPI_Gatherv function later.
    std::vector<int> displs(handler.numOfProcs), recivedCount(handler.numOfProcs);

    // The master process reads the input image.
    if (handler.rank == MASTER) {
        // Reads the input image from a file.
        image1D = h2o::Image::from_file("assets/9.jpg");
        // Stores the number of columns and rows in the input image.
        imgColumns = image1D->columns();
        imgRows = image1D->rows();
        // Prints the number of columns and rows in the input image.
        printf("Image Info: imgColumns:%d, imgRows:%d\n", imgColumns, imgRows);
    }
    // Broadcasts the number of rows and columns in the input image to all processes.
    MPI_Bcast(&imgRows, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&imgColumns, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    // Initializes the convolved image with the same number of rows and columns as the input image.
    convolvedImage = h2o::Image::empty(imgRows, imgColumns);
    
    // If the current process is not the master, it initializes its image1D object as an empty image
    if (handler.rank != MASTER) {
        image1D = h2o::Image::empty(imgRows, imgColumns);
    }
    // Broadcasts the data of the input image to all processes.
    MPI_Bcast(
            image1D->mut_data(),
            imgRows * imgColumns,
            MPI_UNSIGNED_CHAR,
            MASTER,
            MPI_COMM_WORLD
    );
    // Calculates the number of rows that each process will be responsible for processing.
    int offset = imgRows / handler.numOfProcs;
    // Calculates the starting row for the current process.
    int rowStart = handler.rank * offset;
    // Calculates the ending row for the current process.
    int rowEnd = rowStart + offset;

    //If the current process is the master process, it receives additional rows to process.
    if (handler.rank == MASTER) {

        rowEnd = rowEnd + imgRows % handler.numOfProcs;
    }
    // If the current process is the master process, it sets up the displs and recivedCount vectors.
    //for use in the MPI_Gatherv function later.
    if (handler.rank == MASTER) {

        for (int i = 0; i < handler.numOfProcs; i++) {
            if (i == MASTER) {
                // The master process receives the rows assigned to it plus any additional rows.
                recivedCount[i] = (rowEnd - rowStart) * imgColumns;
            } else {
                 // All other processes only receive the rows assigned to them.
                recivedCount[i] = (rowEnd - (imgRows % handler.numOfProcs) - rowStart) * imgColumns;
            }
            // Calculates the displacement for the current process in the final output image.
            displs[i] = (i == 0) ? 0 : displs[i - 1] + recivedCount[i - 1];
        }
    }
    // Creates a ThreadArgs struct to store the arguments to be passed to the Sobel function.
    auto args = ThreadArgs{
            .image = image1D,  // The input image.
            .rowStart = rowStart, // The starting row for the current process.
            .rowEnd = rowEnd, // The ending row for the current process.
    };
    // Creates a std::promise object for storing the output image of the Sobel function.
    auto convolvedImagePartPromise = std::promise<h2o::Image::Ptr>{};
    // Creates a std::future object for retrieving the output image from the std::promise object.
    auto convolvedImagePartFuture = convolvedImagePartPromise.get_future();

    std::jthread thread{
            h2o::sobel,
            args,
            std::move(convolvedImagePartPromise)
    };
    // Waits for the Sobel function to finish executing.
    convolvedImagePartFuture.wait();
    // Retrieves the output image from the std::future object.
    auto convolvedImagePart = convolvedImagePartFuture.get();
    // Gathers the output images from all processes and combines them into the final output image.
    MPI_Gatherv(
            convolvedImagePart->span().data(),
            (rowEnd - rowStart) * imgColumns,
            MPI_UNSIGNED_CHAR,
            convolvedImage->mut_data(),
            recivedCount.data(),
            displs.data(),
            MPI_UNSIGNED_CHAR,
            MASTER,
            MPI_COMM_WORLD
    );
    // If the current process is the master process, it writes the final output image to a file.
    if (handler.rank == MASTER) {
        // Writes the output image to a file in the PNG format.
        stbi_write_png(
                "assets/OUT.png",
                imgColumns,
                imgRows,
                desiredChannels,
                convolvedImage->span().data(),
                imgColumns * desiredChannels
        );
    }
}


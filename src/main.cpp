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

    int imgColumns{}, imgRows{};
    int desiredChannels = 1;

    //For MPI_Gatherv.
    std::vector<int> displs(handler.numOfProcs), recivedCount(handler.numOfProcs);

    //Read image by master proc.
    if (handler.rank == MASTER) {

        image1D = h2o::Image::from_file("assets/9.jpg");

        imgColumns = image1D->columns();
        imgRows = image1D->rows();

        printf("Image Info: imgColumns:%d, imgRows:%d\n", imgColumns, imgRows);
    }

    MPI_Bcast(&imgRows, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&imgColumns, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    convolvedImage = h2o::Image::empty(imgRows, imgColumns);

    if (handler.rank != MASTER) {
        image1D = h2o::Image::empty(imgRows, imgColumns);
    }

    MPI_Bcast(
            image1D->mut_data(),
            imgRows * imgColumns,
            MPI_UNSIGNED_CHAR,
            MASTER,
            MPI_COMM_WORLD
    );

    int offset = imgRows / handler.numOfProcs;
    int rowStart = handler.rank * offset;
    int rowEnd = rowStart + offset;

    //Add extra rows to the master portion
    if (handler.rank == MASTER) {

        rowEnd = rowEnd + imgRows % handler.numOfProcs;
    }

    //For MPI_Gatherv.
    if (handler.rank == MASTER) {

        for (int i = 0; i < handler.numOfProcs; i++) {
            if (i == MASTER) {
                recivedCount[i] = (rowEnd - rowStart) * imgColumns;
            } else {
                recivedCount[i] = (rowEnd - (imgRows % handler.numOfProcs) - rowStart) * imgColumns;
            }

            displs[i] = (i == 0) ? 0 : displs[i - 1] + recivedCount[i - 1];
        }
    }

    auto args = ThreadArgs{
            .image = image1D,
            .rowStart = rowStart,
            .rowEnd = rowEnd,
    };

    auto convolvedImagePartPromise = std::promise<h2o::Image::Ptr>{};
    auto convolvedImagePartFuture = convolvedImagePartPromise.get_future();

    std::jthread thread{
            h2o::sobel,
            args,
            std::move(convolvedImagePartPromise)
    };
    convolvedImagePartFuture.wait();
    auto convolvedImagePart = convolvedImagePartFuture.get();

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

    if (handler.rank == MASTER) {
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


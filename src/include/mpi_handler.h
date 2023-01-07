//
// Created by mythi on 20/12/22.
//

#pragma once // is a preprocessor directive that tells the compiler to include a header file only once in a single translation unit.

#include <mpi.h>

namespace h2o {
    struct MPIHandler {
        int rank;
        int numOfProcs;

        explicit MPIHandler(int argc, char *argv[]) : rank(0), numOfProcs(0) {

            MPI_Init(&argc, &argv);
            MPI_Comm_size(MPI_COMM_WORLD, &numOfProcs);
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        }
        // It is defined to call the 'MPI_Finalize()'
        ~MPIHandler() {

            MPI_Finalize();
        }
    };

}

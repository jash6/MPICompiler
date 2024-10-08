#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// Generated C code from loma for identity function
typedef struct {
    float val;
    float dval;
} _dfloat;

float identity(float x);
_dfloat d_identity(_dfloat x);
_dfloat make__dfloat(float val, float dval);

float identity(float x) {
    return x;
}

_dfloat d_identity(_dfloat x) {
    return make__dfloat(x.val, x.dval);
}

_dfloat make__dfloat(float val, float dval) {
    _dfloat ret;
    ret.val = val;
    ret.dval = dval;
    return ret;
}

// Function to run MPI logic and collect results
_dfloat* run_mpi_logic(int argc, char** argv, int* size) {
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, size);

    // Variable to store the result from each node
    _dfloat local_result;
    _dfloat d_x = {rank, 0.0};
    local_result = d_identity(d_x);
    printf("Rank %d. d_identity Output - val: %f dval: %f\n", rank, local_result.val, local_result.dval);

    // Array to collect results from all nodes
    _dfloat* all_results = NULL;
    if (rank == 0) {
        all_results = (_dfloat*)malloc((*size) * sizeof(_dfloat));
    }

    // Gather results from all nodes to the root node
    MPI_Gather(&local_result, 2, MPI_FLOAT, all_results, 2, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Finalize MPI
    MPI_Finalize();

    return all_results;
}

// Using a main function for now. Ideally, this should be called in Python by the lib file generated by loma.
int main(int argc, char** argv) {
    // Print the number of arguments
    printf("Number of arguments (argc): %d\n", argc);

    // Loop through each argument and print it
    for (int i = 0; i < argc; i++) {
        printf("Argument %d (argv[%d]): %s\n", i, i, argv[i]);
    }
    
    int size;
    _dfloat* results = run_mpi_logic(argc, argv, &size);

    // Print the collected results in the root node
    if (results != NULL) {
        for (int i = 0; i < size; i++) {
            printf("Node %d: val = %f, dval = %f\n", i, results[i].val, results[i].dval);
        }
        free(results);
    }

    return 0;
}

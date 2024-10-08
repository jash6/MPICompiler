// Additional import needed for MPI
#include <mpi.h>
#include <stdio.h>
#include <math.h>


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
	return make__dfloat((x).val,(x).dval);
}

_dfloat make__dfloat(float val, float dval) {
	_dfloat ret;
	ret.val = 0;
	ret.dval = 0;
	(ret).val = val;
	(ret).dval = dval;
	return ret;
}

// Using a main function for now. Ideally this should be called in python by the lib file generated by loma.
int main(int argc, char** argv) {
	// Print the number of arguments
    printf("Number of arguments (argc): %d\n", argc);

    // Loop through each argument and print it
    for (int i = 0; i < argc; i++) {
        printf("Argument %d (argv[%d]): %s\n", i, i, argv[i]);
    }
	
	// Initialize MPI interface
    MPI_Init(&argc, &argv);

	// Rank corresponds to number of nodes
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//  General C code
	float x = rank;
	float iden = identity(rank);
	printf("Rank %d. identity Output - %f\n", rank, iden);

	// Loma Code
	
	_dfloat d_x = {rank, 0.0};
	_dfloat d_iden = d_identity(d_x);
	printf("Rank %d. d_identity Output - val: %f dval: %f\n", rank, d_iden.val, d_iden.dval);

	// Terminates MPI execution environment
    MPI_Finalize();
    return 0;
}
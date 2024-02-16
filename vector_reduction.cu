#ifdef _WIN32
#define NOMINMAX 
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "vector_reduction_kernel.cu"

#include <cuda_runtime.h>

// int readFile(filename:str, data: array);
int readFile(float*, char* filename);

#include "file_io.h"

#define NUM_ELEMENTS 512

void runTest(int argc, char** argv);

float computeOnDevice(float* h_data, int array_mem_size);

extern "C" 
void computeGold(float* reference, float* idata, const unsigned int len);

int main( int argc, char** argv) 
{
    runTest( argc, argv);
    return EXIT_SUCCESS;
}

void runTest( int argc, char** argv) 
{
    int num_elements = NUM_ELEMENTS;
    int errorM = 0;
	
    const unsigned int array_mem_size = sizeof( float) * num_elements;

    // allocate host memory to store the input data
    float* h_data = (float*) malloc( array_mem_size);

    // * No arguments: Randomly generate input data and compare against the 
    //   host's result.
    // * One argument: Read the input data array from the given file.
    switch(argc-1)
    {      
        // --Make your own--
        case 1:  // One Argument
            errorM = readFile(h_data, argv[1]);
            if(errorM != 1)
            {
                printf("Error reading input file!\n");
                exit(1);
            }
        break;

        
        default:  // No Arguments or one argument
            // initialize the input data on the host to be integer values
            // between 0 and 1000
            for( unsigned int i = 0; i < num_elements; ++i) 
            {
                h_data[i] = floorf(1000*(rand()/(float)RAND_MAX));
            }
        break;  
    }
    // compute reference solution
    float reference = 0.0f;
    computeGold(&reference , h_data, num_elements);
    
    float result = computeOnDevice(h_data, num_elements);

    //unsigned int epsilon = 0.0f;
    float epsilon = 0.0f;
    unsigned int result_regtest = (fabs(result - reference) <= epsilon);
    printf( "Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");
    printf( "device: %f  host: %f\n", result, reference);
    // cleanup memory
    free( h_data);
}

int readFile(float* V, char* filename)
{
	float data_read = NUM_ELEMENTS;
	FILE* input = fopen(filename, "r");
	unsigned i = 0;
	for (i = 0; i < data_read; i++)
		fscanf(input, "%f", &(V[i]));
	return data_read;
}

float computeOnDevice(float* h_data, int num_elements)
{
    float* d_data, *d_final_sum;
    int blocks = ceil(num_elements/(float)BLOCK_SIZE);

    cudaMalloc((void**)&d_data, sizeof(float)*num_elements);
    cudaMalloc((void**)&d_final_sum, sizeof(float));

    cudaMemcpy(d_data, h_data, sizeof(float)*num_elements, cudaMemcpyHostToDevice);
    
    reduction<<<blocks, BLOCK_SIZE>>>(d_data, num_elements, d_final_sum);

    cudaMemcpy(h_data, d_final_sum, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_data);
    cudaFree(d_final_sum);
    return *h_data;
}
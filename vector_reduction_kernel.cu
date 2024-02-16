#ifndef _SCAN_NAIVE_KERNEL_H_
#define _SCAN_NAIVE_KERNEL_H_

#define NUM_ELEMENTS 512
#define BLOCK_SIZE 512
#define TILE_SIZE (BLOCK_SIZE * 2)

__global__ void reduction(float *g_data, int n, float *sum_data)
{
	//__shared__ unsigned int partialSums[TILE_SIZE];
	__shared__ float partialSums[TILE_SIZE];
    // calculate the starting index of current block
    unsigned int blockStart = TILE_SIZE * blockIdx.x;
	
    // check bounds and assign zero if exceeding
    if (blockStart+threadIdx.x < n)
        partialSums[threadIdx.x] = g_data[blockStart+threadIdx.x];
    else
        partialSums[threadIdx.x] = 0;
	
	
    if (blockStart + blockDim.x + threadIdx.x < n)
        partialSums[threadIdx.x + blockDim.x] = g_data[blockStart + blockDim.x + threadIdx.x];
    else
        partialSums[threadIdx.x + blockDim.x] = 0;
    // start summing up array elements with a decreasing stride
    for (unsigned int stride = blockDim.x; stride >= 1; stride >>=1)
	{
        __syncthreads();
        if (threadIdx.x<stride)
            partialSums[threadIdx.x] += partialSums[threadIdx.x + stride];
    }
    //save result in sum (global)
    if (threadIdx.x==0)
        atomicAdd(&sum_data[0] ,partialSums[0]);
}

#endif // #ifndef _SCAN_NAIVE_KERNEL_H_

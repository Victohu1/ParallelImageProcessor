/*
EE 451 Final Project - Parallelizing a Photo Editor
Team members: Alice Gusev, Victor Hui, Joshua Williams

References:
How to read and write bmp files in c/c++
https://elcharolin.wordpress.com/2018/11/28/read-and-write-bmp-files-in-c-c/

Read and write BMP file in C
https://codereview.stackexchange.com/questions/196084/read-and-write-bmp-file-in-c

Getting RGB values for each pixel from a raw image in C
https://stackoverflow.com/questions/1536159/getting-rgb-values-for-each-pixel-from-a-raw-image-in-c
*/

#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BDIMX 32
#define BDIMY 32

/*
Steps to Run:
    1. module purge
    2. module load gcc/8.3.0 cuda/10.1.243
    3. nvcc -O3 -o cuda_image_processor cuda_image_processor.cu
    4. srun -n1 --gres=gpu:1 -t1 ./cuda_image_processor <input file name> <output file name> <kernel arg>
*/

void usage() {
    printf("usage: srun -n1 --gres=gpu:1 -t1 ./cuda_image_processor <input file name> <output file name> <kernel arg> <optional: unrolling>\n");
}

// struct for RGB values of a pixel in the image
typedef struct {
    unsigned char r;
    unsigned char g;
    unsigned char b;
} pixel;

// struct for the bmp file
typedef struct {
    int width;
    int height;
    int padding;
    unsigned char header[54];
    pixel *pixels;
} BMP;

// reads the header from the bmp file input
void readHeader(FILE *fp, BMP *bmp) {
    fread(bmp->header, sizeof(bmp->header), 1, fp);
}

// writes the header to the bmp file output
void writeHeader(FILE *fp, BMP *bmp) {
    fwrite(bmp->header, sizeof(bmp->header), 1, fp);
}

// decodes the header from the bmp file input
void decodeHeader(BMP *bmp) {
    bmp->width = *(int*)&bmp->header[18];
    bmp->height = *(int*)&bmp->header[22];
    bmp->padding = (bmp->width*3) % 4;
    bmp->pixels = (pixel*)malloc( sizeof(pixel) * bmp->height * bmp->width );
}

// frees the header created on the heap for the bmp file
void cleanupHeader(BMP *bmp) {
    free(bmp->pixels);
}

// reads the pixel from the input file
pixel readPixel(FILE *fp) {
    pixel result;
    fread(&result, sizeof(result), 1, fp );
    return result;
}

// writes the pixel to the output file
void writePixel(FILE *fp, pixel p) {
    fwrite(&p, sizeof(p), 1, fp);
}

// reads the padding from the input file
void readPadding(FILE *fp, int padding) {
    for ( int i = 0 ; i < padding ; i++ )
        fgetc(fp);  // we don't care, just burn them
}

// writes the padding for the output file
void writePadding(FILE *fp, int padding) {
    for ( int i = 0 ; i < padding ; i++ )
        fputc(0, fp);  // we don't care, just write zeros
}

// reads the image from the input file and stores it in the pixels array
void readImage(FILE *fp, BMP *bmp) {
    for ( int r = 0 ; r < bmp->height ; r++ ) {
        for ( int c = 0 ; c < bmp->width ; c++ ) {
            bmp->pixels[r * bmp->width + c] = readPixel(fp);
        }
        readPadding(fp, bmp->padding);
    }
}

// writes the image stored in the pixels array into the output file
void writeImage(FILE *fp, BMP *bmp) {
    for ( int r = 0 ; r < bmp->height ; r++ ) {
        for ( int c = 0 ; c < bmp->width ; c++ ) {
            writePixel(fp,bmp->pixels[r * bmp->width + c]);
        }
        writePadding(fp, bmp->padding);
    }
}

// kernel to copy the image
__global__ void copy(pixel *odata, pixel *idata, const int width, const int height, const int unrolling)
{
    int x = blockIdx.x * blockDim.x * unrolling + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < unrolling; i++) {
        if (x + i * blockDim.x < width && y < height) {
            odata[y * width + x + i * blockDim.x] = idata[y * width + x + i * blockDim.x];
        }
    }
}

// kernel to copy the image
__global__ void grayscale(pixel *odata, pixel *idata, const int width, const int height, const int unrolling)
{
    int x = blockIdx.x * blockDim.x * unrolling + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < unrolling; i++) {
        if (x + i * blockDim.x < width && y < height) {
            unsigned char gray =
                    idata[y * width + x + i * blockDim.x].r * 0.3 +
                    idata[y * width + x + i * blockDim.x].g * 0.59 +
                    idata[y * width + x + i * blockDim.x].b * 0.11; //luminosity method

            odata[y * width + x + i * blockDim.x].r = gray;
            odata[y * width + x + i * blockDim.x].g = gray;
            odata[y * width + x + i * blockDim.x].b = gray;
        }
    }
}

// kernel to copy the image
__global__ void invert(pixel *odata, pixel *idata, const int width, const int height, const int unrolling)
{

    int x = blockIdx.x * blockDim.x * unrolling + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < unrolling; i++) {
        if (x + i * blockDim.x < width && y < height) {
            odata[y * width + x + i * blockDim.x].r = 255 - idata[y * width + x + i * blockDim.x].r;
            odata[y * width + x + i * blockDim.x].g = 255 - idata[y * width + x + i * blockDim.x].g;
            odata[y * width + x + i * blockDim.x].b = 255 - idata[y * width + x + i * blockDim.x].b;
        }
    }
}

// kernel to copy the image
__global__ void mirror(pixel *odata, pixel *idata, const int width, const int height, const int unrolling)
{
    int x = blockIdx.x * blockDim.x * unrolling + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < unrolling; i++) {
        if (x + i * blockDim.x < width && y < height) {
            odata[y * width + x + i * blockDim.x] = idata[y * width + width - 1 - x - i * blockDim.x];
        }
    }
}

int main (int argc, char **argv) {
    if (argc < 4 || argc > 5) {
        usage();
        exit(-1);
    }

    // open the input and output files
    FILE *in = fopen(argv[1],"rb");
    FILE *out= fopen(argv[2],"wb");

    // get data from bmp image file
    BMP bmp;
    readHeader(in, &bmp);
    decodeHeader(&bmp);
    readImage(in, &bmp);
    writeHeader(out, &bmp);

    // timers initialization
    double cudaStartWarmup, cudaStartTransfer, cudaStartCompute;
    double cudaTimeWarmup, cudaTimeTransfer, cudaTimeCompute;

    // start the timer for entire cuda process
    cudaStartWarmup = seconds();

    // setup device grid
    int unrolling = 1;
    if (argc == 5) {
        unrolling = atoi(argv[4]);
    }

    dim3 block(BDIMX, BDIMY);
    dim3 grid((bmp.width + block.x * unrolling - 1) / (block.x * unrolling), (bmp.height + block.y - 1) / block.y);
    size_t nBytes = sizeof(pixel) * bmp.width * bmp.height;

    // set up kernel
    int iKernel = atoi(argv[3]); // todo: better user inputs for settings
    void (*kernel)(pixel *, pixel *, int, int, int);
    char *kernelName;

    // switch statement that can be used for multiple kernels
    switch (iKernel) {
        case 0:
            kernel = &copy;
            kernelName = "Copy";
            break;

        case 1:
            kernel = &grayscale;
            kernelName = "Grayscale";
            break;

        case 2:
            kernel = &invert;
            kernelName = "Invert";
            break;

        case 3:
            kernel = &mirror;
            kernelName = "Mirror";
            break;
    }

    // allocate device memory
    pixel *oPixels, *iPixels;
    CHECK(cudaMalloc((pixel**)&oPixels, nBytes));
    CHECK(cudaMalloc((pixel**)&iPixels, nBytes));

    // copy data from host to device
    CHECK(cudaMemcpy(iPixels, bmp.pixels, nBytes, cudaMemcpyHostToDevice));

    // warmup to avoide startup overhead
    kernel<<<grid, block>>>(oPixels, iPixels, bmp.width, bmp.height, unrolling);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    cudaTimeWarmup = seconds() - cudaStartWarmup;

    // copy data from host to device
    cudaStartTransfer = seconds();
    CHECK(cudaMemcpy(iPixels, bmp.pixels, nBytes, cudaMemcpyHostToDevice));
    cudaTimeTransfer = seconds() - cudaStartTransfer;

    // run the kernel and record compute time
    cudaStartCompute = seconds();
    kernel<<<grid, block>>>(oPixels, iPixels, bmp.width, bmp.height, unrolling);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    cudaTimeCompute = seconds() - cudaStartCompute;

    // copy data from device to host
    cudaStartTransfer = seconds();
    CHECK(cudaMemcpy(bmp.pixels, oPixels, nBytes, cudaMemcpyDeviceToHost));
    cudaTimeTransfer += seconds() - cudaStartTransfer;

    // free up the used resources
    CHECK(cudaFree(oPixels));
    CHECK(cudaFree(iPixels));
    CHECK(cudaDeviceReset());

    // print the execution times
    printf("%s Kernel:\n", kernelName);
    if (unrolling > 1) {
        printf("Unrolling: %d\n", unrolling);
    }
    printf("\tWarmup Time = %f sec,\n", cudaTimeWarmup);
    printf("\tData Transfer Time = %f sec,\n", cudaTimeTransfer);
    printf("\tComputation Time = %f sec,\n", cudaTimeCompute);

    writeImage(out, &bmp);
    cleanupHeader(&bmp);

    fclose(in);
    fclose(out);
}

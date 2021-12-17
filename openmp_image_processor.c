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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

/*
Steps to Run:
    1. gcc -O3 -fopenmp -o openmp_image_processor openmp_image_processor.c
    2. srun -c<num threads> ./openmp_image_processor <input file name> <output file name> <filter num> <num threads>
*/

void usage() {
    printf("usage: srun -c<num threads> ./openmp_image_processor <input file name> <output file name> <filter num> <num threads>\n");
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
    pixel *copy_pixels;
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
    bmp->copy_pixels = (pixel*)malloc( sizeof(pixel) * bmp->height * bmp->width );
}

// frees the header created on the heap for the bmp file
void cleanupHeader(BMP *bmp) {
    free(bmp->pixels);
    free(bmp->copy_pixels);
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
            bmp->copy_pixels[r * bmp->width + c] = readPixel(fp);
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

void copy(int num_threads, BMP *bmp){
    omp_set_num_threads(num_threads);

    #pragma omp parallel for
    for ( int r = 0 ; r < bmp->height ; r++ ) {
        for ( int c = 0 ; c < bmp->width ; c++ ) {
            bmp->pixels[r * bmp->width + c] = bmp->copy_pixels[r * bmp->width + c];
        }
    }
}

void grayscale(int num_threads, BMP *bmp){
    omp_set_num_threads(num_threads);

    #pragma omp parallel for
    for ( int r = 0 ; r < bmp->height ; r++ ) {
        for ( int c = 0 ; c < bmp->width ; c++ ) {
            unsigned char gray =
                    bmp->copy_pixels[r * bmp->width + c].r * 0.3 +
                    bmp->copy_pixels[r * bmp->width + c].g * 0.59 +
                    bmp->copy_pixels[r * bmp->width + c].b * 0.11; //luminosity method

            bmp->pixels[r * bmp->width + c].r = gray;
            bmp->pixels[r * bmp->width + c].g = gray;
            bmp->pixels[r * bmp->width + c].b = gray;
        }
    }
}

void invert(int num_threads, BMP *bmp){
    omp_set_num_threads(num_threads);

    #pragma omp parallel for
    for ( int r = 0 ; r < bmp->height ; r++ ) {
        for ( int c = 0 ; c < bmp->width ; c++ ) {
            bmp->pixels[r * bmp->width + c].r = 255-bmp->copy_pixels[r * bmp->width + c].r;
            bmp->pixels[r * bmp->width + c].g = 255-bmp->copy_pixels[r * bmp->width + c].g;
            bmp->pixels[r * bmp->width + c].b = 255-bmp->copy_pixels[r * bmp->width + c].b;
        }
    }
}

void mirror(int num_threads, BMP *bmp){
    omp_set_num_threads(num_threads);

    #pragma omp parallel for
    for ( int r = 0 ; r < bmp->height ; r++ ) {
        for ( int c = 0 ; c < bmp->width ; c++ ) {
            bmp->pixels[r * bmp->width + c] = bmp->copy_pixels[r * bmp->width + bmp->width - 1 - c];
        }
    }
}

int main (int argc, char **argv) {
    if (argc != 5) {
        usage();
        exit(-1);
    }

    // open the input and output files
    FILE *in = fopen(argv[1], "rb");
    FILE *out= fopen(argv[2], "wb");

    // get the user inputs for filter and number of threads
    int filter = atoi(argv[3]);
    int num_threads = atoi(argv[4]);

    // get data from bmp image file
    BMP bmp;
    readHeader(in, &bmp);
    decodeHeader(&bmp);
    readImage(in, &bmp);
    writeHeader(out, &bmp);

    struct timespec start, stop;
    double time;

    if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}

    if (filter == 0) {
        copy(num_threads, &bmp);
    } else if (filter == 1) {
        grayscale(num_threads, &bmp);
    } else if (filter == 2) {
        invert(num_threads, &bmp);
    } else if (filter == 3) {
        mirror(num_threads, &bmp);
    }

    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}
    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;

    if (filter == 0) {
        printf("Copy filter:\n");
    } else if (filter == 1) {
        printf("Grayscale filter:\n");
    } else if (filter == 2) {
        printf("Inverting filter:\n");
    } else if (filter == 3) {
        printf("Mirroring filter:\n");
    }

    printf("\tNumber of threads = %d\n", num_threads);
    printf("\tExecution time = %f seconds\n", time);

    writeImage(out, &bmp);
    cleanupHeader(&bmp);

    fclose(in);
    fclose(out);
}

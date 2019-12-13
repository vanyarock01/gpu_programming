#include <stdio.h>
#include <string.h>
#include <math.h>

#ifndef CSC_H
#define CSC_H

#define CSC(call)                   \
do {                                \
    cudaError_t res = call;         \
    if (res != cudaSuccess) {       \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));       \
        exit(0);                    \
    }                               \
} while(0)
#endif

#define VEC_DIM 3

#define AXIS_X  0
#define AXIS_Y  1
#define AXIS_Z  2

#define CPU true

#ifndef CPU
    #define NEG_INF_F __int_as_float(0xff800000)
#else
    #define NEG_INF_F -INFINITY
#endif

typedef struct _vec_3x1
{
    double data[VEC_DIM];
} vec_3x1;

typedef struct _image
{
    int w;
    int h;
    uchar4 *data;
} image;

typedef struct _pixel_class
{
    int size;
    int *x, *y;
    vec_3x1 avg;

} pixel_class;

__constant__ vec_3x1 AVG[32]; // constant variable in GPU memory
vec_3x1 AVG_SIMPLE[32];       // simple variable

image*
read_image(const char *filename)
{
    image *img = (image *) malloc(sizeof(image));

    FILE *in = fopen(filename, "rb");
    fread(&img->w, sizeof(img->w), 1, in);
    fread(&img->h, sizeof(img->h), 1, in); 
    img->data = (uchar4 *) malloc(sizeof(uchar4) * img->w * img->h);
    fread(img->data, sizeof(img->data), img->w * img->h, in);
    fclose(in);

    return img;
}

void
write_image(const char *filename, const image *img)
{
    FILE *out = fopen(filename, "wb");
    fwrite(&img->w, sizeof(img->w), 1, out);
    fwrite(&img->h, sizeof(img->h), 1, out);
    fwrite(img->data, sizeof(img->data), img->w * img->h, out);
    
    fclose(out);
}

__host__ __device__ int
pos_from_2D(int x, int y, int w)
{
    return w * y + x;
}

pixel_class *
read_pixel_classes(const int nc, const image *img)
{
    pixel_class *c = (pixel_class *) malloc(sizeof(pixel_class) * nc);
    for (int i = 0; i < nc; i++) {
        int np;
        scanf("%d", &np);

        c[i].x = (int *) malloc(sizeof(int) * np);
        c[i].y = (int *) malloc(sizeof(int) * np);

        memset((c[i].avg).data, 0.0, sizeof((c[i].avg).data));
        for (int j = 0; j < np; j++) {
            scanf("%d %d", &(c[i].x[j]), &(c[i].y[j]));

            uchar4 p = img->data[pos_from_2D(c[i].x[j], c[i].y[j], img->w)];
            
            c[i].avg.data[AXIS_X] += p.x;
            c[i].avg.data[AXIS_Y] += p.y;
            c[i].avg.data[AXIS_Z] += p.z;
        }
        c[i].avg.data[AXIS_X] /= (double) np;
        c[i].avg.data[AXIS_Y] /= (double) np;
        c[i].avg.data[AXIS_Z] /= (double) np;
    }
    return c;
}

void
delete_pixel_classes(pixel_class *c)
{
    free(c->x);
    free(c->y);
    free(c);
    c = NULL;
}

void
delete_image(image *img)
{
    free(img->data);
    free(img);
    img = NULL;
}

__device__ double
calc_min_dst(const uchar4 *p, int idx)
{
    vec_3x1 val;

    val.data[AXIS_X] = p->x - AVG[idx].data[AXIS_X];
    val.data[AXIS_Y] = p->y - AVG[idx].data[AXIS_Y];
    val.data[AXIS_Z] = p->z - AVG[idx].data[AXIS_Z];


    double dst = 0.0f;
    for (int i = 0; i < VEC_DIM; i++) {
        dst += val.data[i] * val.data[i];
    }

    return -dst;
}


__global__ void
kernel_min_dst(uchar4 *dst, int w, int h, int nc)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = gridDim.x * blockDim.x;
    int offsety = gridDim.y * blockDim.y;

    for (int x = idx; x < w; x += offsetx) {
        for (int y = idy; y < h; y += offsety) {
            int pos = pos_from_2D(x, y, w);

            double max_dst = NEG_INF_F;
            int    max_idx = 0;

            for (int k = 0; k < nc; k++) {
                double distance = calc_min_dst(&dst[pos], k);
                if ( distance > max_dst ) {
                    max_dst = distance  ;
                    max_idx = k;
                }
            }
            dst[pos].w = max_idx;
        }
    }
}

double
calc_min_dst_CPU(const uchar4 *p, int idx)
{
    vec_3x1 val;

    val.data[AXIS_X] = p->x - AVG_SIMPLE[idx].data[AXIS_X];
    val.data[AXIS_Y] = p->y - AVG_SIMPLE[idx].data[AXIS_Y];
    val.data[AXIS_Z] = p->z - AVG_SIMPLE[idx].data[AXIS_Z];

    double dst = 0.0f;
    for (int i = 0; i < VEC_DIM; i++) {
        dst += val.data[i] * val.data[i];
    }

    return -dst;
}

__host__ void
kernel_min_dst_CPU(uchar4 *dst, int w, int h, int nc)
{
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {
            int pos = pos_from_2D(x, y, w);

            double max_dst = NEG_INF_F;
            int    max_idx = 0;

            for (int k = 0; k < nc; k++) {
                double distance = calc_min_dst_CPU(&dst[pos], k);
                if ( distance > max_dst ) {
                    max_dst = distance;
                    max_idx = k;
                }
            }
            dst[pos].w = max_idx;
        }
    }
}

#ifndef CPU
int
main (int argc, char const *argv[])
{
    char src_file[256];
    char dst_file[256];
    scanf("%s", src_file);
    scanf("%s", dst_file);

    image *img = read_image(src_file);

    int nc;
    scanf("%d", &nc);
    pixel_class *c = read_pixel_classes(nc, img);
    
    uchar4* dst;
    CSC(cudaMalloc(&dst, sizeof(uchar4) * img->w * img->h));
    CSC(cudaMemcpy(dst, img->data, sizeof(uchar4) * img->w * img->h, cudaMemcpyHostToDevice));

    for (int i = 0; i < nc; i++) {
        CSC(cudaMemcpyToSymbol(AVG, &(c[i].avg), sizeof(vec_3x1), i * sizeof(vec_3x1)));
    }
    
    dim3 gridSize(16, 16);
    dim3 blockSize(16, 16);
    
    kernel_min_dst <<<gridSize, blockSize>>> (dst, img->w, img->h, nc);
    CSC(cudaGetLastError());
    
    CSC(cudaMemcpy(img->data, dst, sizeof(uchar4) * img->w * img->h, cudaMemcpyDeviceToHost));
    write_image(dst_file, img);
    
    delete_pixel_classes(c);
    delete_image(img);
    return 0;
}
#else
int
main (int argc, char const *argv[])
{
    char src_file[256];
    char dst_file[256];
    scanf("%s", src_file);
    scanf("%s", dst_file);

    image *img = read_image(src_file);

    int nc;
    scanf("%d", &nc);
    pixel_class *c = read_pixel_classes(nc, img);
    
    uchar4 *dst = (uchar4 *) malloc(sizeof(uchar4) * img->w * img->h);

    memcpy(dst, img->data, sizeof(uchar4) * img->w * img->h);

    for (int i = 0; i < nc; i++) {
        memcpy(AVG_SIMPLE[i].data, c[i].avg.data, sizeof(c[i].avg.data));
    }
    
    kernel_min_dst_CPU(dst, img->w, img->h, nc);
    
    memcpy(img->data, dst, sizeof(uchar4) * img->w * img->h);
    write_image(dst_file, img);
    
    delete_pixel_classes(c);    
    delete_image(img);
    return 0;
}
#endif
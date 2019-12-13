#include <stdio.h>

#define CPU true

#define CSC(call) do { \
        cudaError_t res = call; \
        if (res != cudaSuccess) { \
                fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(res)); \
                exit(0); \
        } \
} while (0)

typedef struct _image
{
    int w;
    int h;
    uchar4 *data;
} image;

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

void
delete_image(image *img)
{
    free(img->data);
    free(img);
    img = NULL;
}

texture <uchar4, 2, cudaReadModeElementType> tex;

__device__ __host__ int
pos(int i, int border)
{
    return max(0, min(i, border));
}

__global__ void
gaussian_kernel(
    uchar4 *dst,
    int     radius,
    float   div,
    int     w,
    int     h,
    int     axis_x,
    int     axis_y)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    
    int x, y, c;
    uchar4 p;

    for (x = idx; x < w; x += offsetx) {
        for (y = idy; y < h; y += offsety) {

            float r = 0.0f,
                  g = 0.0f,
                  b = 0.0f;

            float weight = 0.0f;

            for (c = -radius; c <= radius; c++) {
                weight = exp( -(float) (c * c) / (float) (2 * radius * radius));
                int posx = pos(x + (c * axis_x), w);
                int posy = pos(y + (c * axis_y), h);

                p = tex2D(tex, (float) posx, (float) posy);

                r += (p.x) * weight;
                g += (p.y) * weight;
                b += (p.z) * weight;

            }

            dst[x + y * w] = make_uchar4(
                (unsigned char) (r / div),
                (unsigned char) (g / div),
                (unsigned char) (b / div), 0.0f);
        }
    }
}

__host__ void
gaussian_kernel_CPU(
    uchar4 *src,
    uchar4 *dst,
    int     radius,
    float   div,
    int     w,
    int     h,
    int     axis_x,
    int     axis_y)
{

    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {
            float r = 0.0f,
                  g = 0.0f,
                  b = 0.0f;

            float weight = 0.0f;

            for (int c = -radius; c <= radius; c++) {
                weight = exp( -(float) (c * c) / (float) (2 * radius * radius));
                int posx = pos(x + (c * axis_x), w);
                int posy = pos(y + (c * axis_y), h);

                uchar4 p = src[posx + posy * w];

                r += (p.x) * weight;
                g += (p.y) * weight;
                b += (p.z) * weight;

            }

            dst[x + y * w] = make_uchar4(
                (unsigned char) (r / div),
                (unsigned char) (g / div),
                (unsigned char) (b / div), 0.0f);
        }
    }

}
#ifndef CPU
int
main (int argc, char const *argv[])
{
    printf("%s\n", "-- GPU VERSION");

    int  radius;
    char src_file[256];
    char dst_file[256];

    scanf("%s", src_file);
    scanf("%s", dst_file);
    scanf("%d", &radius);

    image *img = read_image(src_file);

    cudaArray *arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, img->w, img->h));
    CSC(cudaMemcpyToArray(arr, 0, 0, img->data, sizeof(uchar4) * img->h * img->w, cudaMemcpyHostToDevice));

    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.channelDesc = ch;
    tex.filterMode = cudaFilterModePoint;
    tex.normalized = false;

    dim3 gridSize  (32, 32);
    dim3 blockSize (32, 32);

    cudaBindTextureToArray(tex, arr, ch);
    uchar4 *dev_data;
    cudaMalloc(&dev_data, sizeof(uchar4) * img->h * img->w);

    if ( radius > 0 ) {
        float div = 0.0f;
        for (int v = -radius; v <= radius; v++) {
            div += exp( -(float) (v * v) / (float) (2 * radius * radius) );
        }

        gaussian_kernel <<<gridSize, blockSize>>>(dev_data, radius, div, img->w, img->h, 1, 0);
        CSC(cudaGetLastError());

        CSC(cudaDeviceSynchronize());

        CSC(cudaMemcpy(img->data, dev_data, sizeof(uchar4) * img->h * img->w, cudaMemcpyDeviceToHost));
        CSC(cudaMemcpyToArray(arr, 0, 0, img->data, sizeof(uchar4) * img->h * img->w, cudaMemcpyHostToDevice));
        
        gaussian_kernel<<<gridSize, blockSize>>>(dev_data, radius, div, img->w, img->h, 0, 1);
        CSC(cudaGetLastError());
        CSC(cudaDeviceSynchronize());
        CSC(cudaMemcpy(img->data, dev_data, sizeof(uchar4) * img->h * img->w, cudaMemcpyDeviceToHost));
    }
    write_image(dst_file, img);

    CSC(cudaUnbindTexture(tex));
    CSC(cudaFreeArray(arr));
    CSC(cudaFree(dev_data));
    delete_image(img);

    return 0;
}
#else // CPU VERSION
int
main(int argc, char const *argv[])
{
    printf("%s\n", "-- CPU VERSION");
    
    int  radius;
    char src_file[256];
    char dst_file[256];

    scanf("%s", src_file);
    scanf("%s", dst_file);
    scanf("%d", &radius);

    image *img = read_image(src_file);

    uchar4 *dev_data = (uchar4 *) malloc (img->w * img->h * sizeof(uchar4));

    if ( radius > 0 ) {
        float div = 0.0f;
        for (int v = -radius; v <= radius; v++) {
            div += exp( -(float) (v * v) / (float) (2 * radius * radius) );
        }
        // asix X - to dev_data
        gaussian_kernel_CPU(img->data, dev_data, radius, div, img->w, img->h, 1, 0);
        // asix Y - to img->data
        gaussian_kernel_CPU(dev_data, img->data, radius, div, img->w, img->h, 0, 1);

    }
    write_image(dst_file, img);
    delete_image(img);

    return 0;
}
#endif
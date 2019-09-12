#include <stdio.h>
#include <stdlib.h>

typedef double vec_item;

vec_item*  safe_malloc  (int n);
vec_item*  read_vector  (int n);
void       print_vector (const vec_item *vec, int n);


__global__ void
sub_vector_kernel(vec_item *c, const vec_item *a, const vec_item *b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] - b[i];
    }
}

vec_item*
sub_vector(const vec_item *a, const vec_item *b, int n)
{
    vec_item *work_a = NULL;
    vec_item *work_b = NULL;
    vec_item *work_c = NULL;

    cudaMalloc( (void **) &work_a, n * sizeof(vec_item) );
    cudaMalloc( (void **) &work_b, n * sizeof(vec_item) );
    cudaMalloc( (void **) &work_c, n * sizeof(vec_item) );

    cudaMemcpy( work_a, a, n * sizeof(vec_item), cudaMemcpyHostToDevice );
    cudaMemcpy( work_b, b, n * sizeof(vec_item), cudaMemcpyHostToDevice );

    sub_vector_kernel <<<2, (n + 1) / 2>>> (work_c, work_a, work_b, n);

    cudaDeviceSynchronize();

    vec_item *c = (vec_item *) malloc( n * sizeof(vec_item) );

    cudaMemcpy(c, work_c, n * sizeof(vec_item), cudaMemcpyDeviceToHost);

    cudaFree(work_a);
    cudaFree(work_b);
    cudaFree(work_c);

    return c;
}

int
main(int argc, char const *argv[])
{
    int n;
    scanf("%d", &n);

    
    vec_item *a = read_vector(n);
    vec_item *b = read_vector(n);

    vec_item *c = sub_vector(a, b, n);

    print_vector(c, n);

    free(a);
    free(b);
    free(c);

    return 0;
}

vec_item*
safe_malloc(int n)
{
    vec_item *vec = (vec_item *) malloc( n * sizeof(vec_item) );
    if (vec == NULL) {
        fprintf(stderr, "ERROR: unable to allocate %d bytes.\n", n);
        exit(0);
    }
    return vec;
}


vec_item*
read_vector(int n)
{
    vec_item *vec = safe_malloc(n);

    for (int i = 0; i < n; i++) {
        scanf("%lf", vec + i);
    }
    return vec;
}

void
print_vector(const vec_item *vec, int n)
{
    for (size_t i = 0; i < n; i++) {
        printf("%lf ", *(vec + i));
    }
    printf("\n");
}
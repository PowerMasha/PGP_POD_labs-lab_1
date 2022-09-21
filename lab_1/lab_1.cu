#include <stdlib.h> 
#include <stdio.h> 
 
__global__ void SKernel(float *a, float *b, int n) { 
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    int offset = blockDim.x * gridDim.x;
    while(idx < n) {
        b[idx] = sqrt(a[idx]);
        idx += offset;
    } 
} 
 
void Printer(float *a, int n){ 
    for (int i = 0; i < n; i++){ 
        printf("%f\n", a[i]); 
    } 
} 
 
void Assigner(float *a, int n){ 
    for (int i = 0; i < n; i++){ 
        a[i] = (float)i; 
    } 
} 
 
int main() { 
    
    int n = 100; 
    int size = n * sizeof(float);
    
    float *aDev = NULL, *bDev = NULL; 
    float *a = NULL, *b = NULL; 
    
    cudaMalloc((void **) &aDev, size); 
    cudaMalloc((void **) &bDev, size); 
    
    a = (float *) malloc(size); 
    b = (float *) malloc(size); 
    
    Assigner(a, n); 

    cudaMemcpy(aDev, a, size, cudaMemcpyHostToDevice); 
    cudaMemcpy(bDev, b, size, cudaMemcpyHostToDevice); 
    
    SKernel<<<256, 256>>> (aDev, bDev, n); 
    
    cudaMemcpy(b, bDev, size, cudaMemcpyDeviceToHost); 
    
    Printer(b, n); 
    
    cudaFree(aDev); 
    cudaFree(bDev); 
    
    free(a); 
    free(b); 
}
#include <cuda_runtime.h>
#include "core.h"  // Shared header
#include "parameters.h"

#define ECUT (4.0f * (powf(RCUT, -12.0f) - powf(RCUT, -6.0f)))

__device__ float3 minimum_image(float3 rij, float L) {
    rij.x -= rintf(rij.x / L) * L;
    rij.y -= rintf(rij.y / L) * L;
    rij.z -= rintf(rij.z / L) * L;
    return rij;
}

__global__ void forces_kernel(const float4* rxyz, float4* fxyz, 
        float* epot, float* pres_vir,
        float rcut2, float L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= N || j >= N || i >= j) return;  // Solo pares i < j

    float4 ri = rxyz[i];
    float4 rj = rxyz[j];
    float3 rij = {ri.x - rj.x, ri.y - rj.y, ri.z - rj.z};
    rij = minimum_image(rij, L);

    float rij2 = rij.x*rij.x + rij.y*rij.y + rij.z*rij.z;
    if (rij2 <= rcut2) {
        float r2inv = 1.0f / rij2;
        float r6inv = r2inv * r2inv * r2inv;
        float fr = 24.0f * r2inv * r6inv * (2.0f * r6inv - 1.0f);

        // Fuerzas
        float3 force = {fr * rij.x, fr * rij.y, fr * rij.z};

        atomicAdd(&fxyz[i].x,  force.x);
        atomicAdd(&fxyz[i].y,  force.y);
        atomicAdd(&fxyz[i].z,  force.z);

        atomicAdd(&fxyz[j].x, -force.x);
        atomicAdd(&fxyz[j].y, -force.y);
        atomicAdd(&fxyz[j].z, -force.z);

        float pair_epot = 4.0f * r6inv * (r6inv - 1.0f) - ECUT;
        float pair_pres = fr * rij2;

        atomicAdd(epot, pair_epot);
        atomicAdd(pres_vir, pair_pres);
    }
}

void forces(const float* rxyz, float* fxyz, float* epot, float* pres,
        const float* temp, float rho, float V, float L) {
    float *d_rxyz, *d_fxyz, *d_epot, *d_pres_vir;
    size_t size = 4 * N * sizeof(float);

    cudaMalloc(&d_rxyz, size);
    cudaMalloc(&d_fxyz, size);
    cudaMalloc(&d_epot, sizeof(float));
    cudaMalloc(&d_pres_vir, sizeof(float));

    cudaMemcpy(d_rxyz, rxyz, size, cudaMemcpyHostToDevice);
    cudaMemset(d_fxyz, 0, size);
    cudaMemset(d_epot, 0, sizeof(float));
    cudaMemset(d_pres_vir, 0, sizeof(float));

    int threads_per_block = 16;  // Bloques 16x16 = 256 hilos
    dim3 blocks((N + threads_per_block - 1) / threads_per_block,
            (N + threads_per_block - 1) / threads_per_block);
    dim3 threads(threads_per_block, threads_per_block);
    forces_kernel<<<blocks, threads>>>((float4*)d_rxyz, (float4*)d_fxyz, 
            d_epot, d_pres_vir, RCUT*RCUT, L);

    cudaMemcpy(fxyz, d_fxyz, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(epot, d_epot, sizeof(float), cudaMemcpyDeviceToHost);

    float h_pres_vir;
    cudaMemcpy(&h_pres_vir, d_pres_vir, sizeof(float), cudaMemcpyDeviceToHost);
    *pres = *temp * rho + h_pres_vir / (3.0f * V);

    cudaFree(d_rxyz);
    cudaFree(d_fxyz);
    cudaFree(d_epot);
    cudaFree(d_pres_vir);
}

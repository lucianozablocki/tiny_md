#include "core.h"
#include "parameters.h"

#include <math.h>
#include <immintrin.h>
#include "mtwister.h"
#include <stdlib.h>

#define ECUT (4.0f * (powf(RCUT, -12.0f) - powf(RCUT, -6.0f)))

void init_pos(float* rxyz, const float rho)
{
    float a = cbrtf(4.0f / rho);
    int nucells = roundf(cbrtf((float)N / 4.0f));
    int idx = 0;

    for (int i = 0; i < nucells; i++) {
        for (int j = 0; j < nucells; j++) {
            for (int k = 0; k < nucells; k++) {
                rxyz[idx + 0] = i * a;
                rxyz[idx + 1] = j * a;
                rxyz[idx + 2] = k * a;
                rxyz[idx + 3] = 0.0f;

                rxyz[idx + 4] = (i + 0.5f) * a;
                rxyz[idx + 5] = (j + 0.5f) * a;
                rxyz[idx + 6] = k * a;
                rxyz[idx + 7] = 0.0f;

                rxyz[idx + 8] = (i + 0.5f) * a;
                rxyz[idx + 9] = j * a;
                rxyz[idx + 10] = (k + 0.5f) * a;
                rxyz[idx + 11] = 0.0f;

                rxyz[idx + 12] = i * a;
                rxyz[idx + 13] = (j + 0.5f) * a;
                rxyz[idx + 14] = (k + 0.5f) * a;
                rxyz[idx + 15] = 0.0f;
                idx += 16;
            }
        }
    }
}


void init_vel(float* vxyz, float* temp, float* ekin, MTRand* r)
{
    float sf, sumvx = 0.0f, sumvy = 0.0f, sumvz = 0.0f, sumvw = 0.0f, sumv2 = 0.0f;

    for (int i = 0; i < 4 * N; i += 4) {
        vxyz[i + 0] = genRand(r) - 0.5f;
        vxyz[i + 1] = genRand(r) - 0.5f;
        vxyz[i + 2] = genRand(r) - 0.5f;
        vxyz[i + 3] = 0.0f;
    }

    for (int i = 0; i < 4 * N; i += 4) {
        sumvx += vxyz[i + 0];
        sumvy += vxyz[i + 1];
        sumvz += vxyz[i + 2];
        sumvw += vxyz[i + 3];
    }

    for (int i = 0; i < 4 * N; i += 4) {
        sumv2 += vxyz[i + 0] * vxyz[i + 0] + vxyz[i + 1] * vxyz[i + 1] +
            vxyz[i + 2] * vxyz[i + 2] + vxyz[i + 3] * vxyz[i + 3];
    }

    sumvx /= (float)N;
    sumvy /= (float)N;
    sumvz /= (float)N;
    sumvw /= (float)N;
    *temp = sumv2 / (3.0f * N);
    *ekin = 0.5f * sumv2;
    sf = sqrtf(T0 / *temp);

    for (int i = 0; i < 4 * N; i += 4) {
        vxyz[i + 0] = (vxyz[i + 0] - sumvx) * sf;
        vxyz[i + 1] = (vxyz[i + 1] - sumvy) * sf;
        vxyz[i + 2] = (vxyz[i + 2] - sumvz) * sf;
        vxyz[i + 3] = (vxyz[i + 3] - sumvw) * sf;
    }
}

static inline __m256 pbc_avx(__m256 coords, const float cell_length)
{
    __m256 full_cell_length = _mm256_set1_ps(cell_length);

    // mask_lt: coords <= 0
    __m256 mask_lt = _mm256_cmp_ps(coords, _mm256_setzero_ps(), _CMP_LE_OQ);

    // mask_gt: coords > cell_length
    __m256 mask_gt = _mm256_cmp_ps(coords, full_cell_length, _CMP_GT_OQ);

    // compute addition and subtraction branches
    __m256 add_cell_length = _mm256_add_ps(coords, full_cell_length);
    __m256 sub_cell_length = _mm256_sub_ps(coords, full_cell_length);

    // Apply masks
    __m256 coords_tmp = _mm256_blendv_ps(coords, add_cell_length, mask_lt);
    return _mm256_blendv_ps(coords_tmp, sub_cell_length, mask_gt);
}

void velocity_verlet(float* restrict rxyz, float* restrict vxyz, float* restrict fxyz, float* epot,
        float* ekin, float* pres, float* temp, const float rho,
        const float V, const float L)
{

    for (int i = 0; i < 4 * N; i += 4) { // actualizo posiciones
        rxyz[i + 0] += vxyz[i + 0] * DT + 0.5f * fxyz[i + 0] * DT * DT;
        rxyz[i + 1] += vxyz[i + 1] * DT + 0.5f * fxyz[i + 1] * DT * DT;
        rxyz[i + 2] += vxyz[i + 2] * DT + 0.5f * fxyz[i + 2] * DT * DT;
        rxyz[i + 3] += vxyz[i + 3] * DT + 0.5f * fxyz[i + 3] * DT * DT;
    }

    for (int i = 0; i < 4 * N; i += 4) { // actualizo posiciones
        vxyz[i + 0] += 0.5f * fxyz[i + 0] * DT;
        vxyz[i + 1] += 0.5f * fxyz[i + 1] * DT;
        vxyz[i + 2] += 0.5f * fxyz[i + 2] * DT;
        vxyz[i + 3] += 0.5f * fxyz[i + 3] * DT;
    }

    
    for (int i = 0; i < 4 * N; i += 8) { // actualizo posiciones
        __m256 r = _mm256_loadu_ps(rxyz + i);
        r = pbc_avx(r, L);
        _mm256_storeu_ps(rxyz + i, r);
    }

    forces(rxyz, fxyz, epot, pres, temp, rho, V, L); // actualizo fuerzas

    float sumv2 = 0.0f;
    for (int i = 0; i < 4 * N; i += 4) { // actualizo velocidades
        vxyz[i + 0] += 0.5f * fxyz[i + 0] * DT;
        vxyz[i + 1] += 0.5f * fxyz[i + 1] * DT;
        vxyz[i + 2] += 0.5f * fxyz[i + 2] * DT;
        vxyz[i + 3] += 0.5f * fxyz[i + 3] * DT;
    }

    for (int i = 0; i < 4 * N; i += 4) { // actualizo velocidades
        sumv2 += vxyz[i + 0] * vxyz[i + 0] + vxyz[i + 1] * vxyz[i + 1]
            + vxyz[i + 2] * vxyz[i + 2] + vxyz[i + 3] * vxyz[i + 3];
    }

    *ekin = 0.5f * sumv2;
    *temp = sumv2 / (3.0f * N);
}

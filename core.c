#include "core.h"
#include "parameters.h"

#include <math.h>
#include <stdlib.h> // rand()
#include <immintrin.h>

#define ECUT (4.0 * (pow(RCUT, -12) - pow(RCUT, -6)))


void init_pos(double* rxyz, const double rho)
{
    // inicialización de las posiciones de los átomos en un cristal FCC

    double a = cbrt(4.0 / rho);
    int nucells = round(cbrt((double)N / 4.0));
    int idx = 0;

    for (int i = 0; i < nucells; i++) {
        for (int j = 0; j < nucells; j++) {
            for (int k = 0; k < nucells; k++) {
                rxyz[idx + 0] = i * a; // x
                rxyz[idx + 1] = j * a; // y
                rxyz[idx + 2] = k * a; // z
                rxyz[idx + 3] = 0; // z
                                   // del mismo átomo
                rxyz[idx + 4] = (i + 0.5) * a;
                rxyz[idx + 5] = (j + 0.5) * a;
                rxyz[idx + 6] = k * a;
                rxyz[idx + 7] = 0; // z

                rxyz[idx + 8] = (i + 0.5) * a;
                rxyz[idx + 9] = j * a;
                rxyz[idx + 10] = (k + 0.5) * a;
                rxyz[idx + 11] = 0; // z

                rxyz[idx + 12] = i * a;
                rxyz[idx + 13] = (j + 0.5) * a;
                rxyz[idx + 14] = (k + 0.5) * a;
                rxyz[idx + 15] = 0; // z

                idx += 16;
            }
        }
    }
}


void init_vel(double* vxyz, double* temp, double* ekin)
{
    // inicialización de velocidades aleatorias

    double sf, sumvx = 0.0, sumvy = 0.0, sumvz = 0.0, sumvw = 0.0, sumv2 = 0.0;

    for (int i = 0; i < 4 * N; i += 4) {
        vxyz[i + 0] = rand() / (double)RAND_MAX - 0.5;
        vxyz[i + 1] = rand() / (double)RAND_MAX - 0.5;
        vxyz[i + 2] = rand() / (double)RAND_MAX - 0.5;
        vxyz[i + 3] = 0;
    }

    for (int i = 0; i < 4 * N; i += 4) {
        sumvx += vxyz[i + 0];
        sumvy += vxyz[i + 1];
        sumvz += vxyz[i + 2];
        sumvw += vxyz[i + 3];
    }

    for (int i = 0; i < 4 * N; i += 4) {
        sumv2 += vxyz[i + 0] * vxyz[i + 0] + vxyz[i + 1] * vxyz[i + 1]
            + vxyz[i + 2] * vxyz[i + 3] + vxyz[i + 3] * vxyz[i + 3];
    }

    sumvx /= (double)N;
    sumvy /= (double)N;
    sumvz /= (double)N;
    sumvw /= (double)N;
    *temp = sumv2 / (3.0 * N);
    *ekin = 0.5 * sumv2;
    sf = sqrt(T0 / *temp);

    for (int i = 0; i < 4 * N; i += 4) { // elimina la velocidad del centro de masa
                                         // y ajusta la temperatura
        vxyz[i + 0] = (vxyz[i + 0] - sumvx) * sf;
        vxyz[i + 1] = (vxyz[i + 1] - sumvy) * sf;
        vxyz[i + 2] = (vxyz[i + 2] - sumvz) * sf;
        vxyz[i + 3] = (vxyz[i + 3] - sumvw) * sf;
    }
}


static inline double minimum_image(double cordi, const double cell_length)
{
    // imagen más cercana

    if (cordi <= -0.5 * cell_length) {
        cordi += cell_length;
    } else if (cordi > 0.5 * cell_length) {
        cordi -= cell_length;
    }
    return cordi;
}

static inline __m256d minimum_image_avx(__m256d cords, double cell_length) {
    __m256d half_cell_length = _mm256_set1_pd(0.5 * cell_length);
    __m256d full_cell_length = _mm256_set1_pd(cell_length);

    // mask_lt: cords <= -0.5 * cell_length
    __m256d minus_half_cell_length = _mm256_sub_pd(_mm256_setzero_pd(), half_cell_length);
    __m256d mask_lt = _mm256_cmp_pd(cords, minus_half_cell_length, _CMP_LE_OQ);

    // mask_gt: cords > 0.5 * cell_length
    __m256d mask_gt = _mm256_cmp_pd(cords, half_cell_length, _CMP_GT_OQ);

    // Apply cocordscordsections
    __m256d add_cell_length = _mm256_add_pd(cords, full_cell_length);
    __m256d sub_cell_length = _mm256_sub_pd(cords, full_cell_length);

    // Blend cordsesults
    __m256d cords_tmp = _mm256_blendv_pd(cords, add_cell_length, mask_lt);
    return _mm256_blendv_pd(cords_tmp, sub_cell_length, mask_gt);
}


void forces(const double* restrict rxyz, double* restrict fxyz, double* restrict epot, double* restrict pres,
        const double* restrict temp, const double rho, const double V, const double L)
{
    // calcula las fuerzas LJ (12-6)

    for (int i = 0; i < 4 * N; i++) {
        fxyz[i] = 0.0;
    }
    double pres_vir = 0.0;
    double rcut2 = RCUT * RCUT;
    *epot = 0.0;

    for (int i = 0; i < 4 * (N - 1); i += 4) {

        __m256d ri = _mm256_loadu_pd(rxyz + i);

        for (int j = i + 4; j < 4 * N; j += 4) {

            __m256d rj = _mm256_loadu_pd(rxyz + j);

            // rij = ri - rj
            __m256d rij = _mm256_sub_pd(ri, rj);

            rij = minimum_image_avx(rij, L);

            // rij2 = rij * rij
            __m256d rij2 = _mm256_mul_pd(rij, rij); // [da|db|dc|dd]

            __m256d shuf1 = _mm256_permute2f128_pd(rij2, rij2, 1); // [dc|dd|da|db]
            __m256d sum1 = _mm256_add_pd(rij2, shuf1); // [da+dc|db+dd|dc+da|dd+db]
            __m128d lo128 = _mm256_castpd256_pd128(sum1); // [da+dc|db+dd]
            __m128d hi128 = _mm_unpackhi_pd(lo128, lo128); // [db+dd|db+dd]
            __m128d total = _mm_add_sd(lo128, hi128); // [da+dc+db+dd|db+dd+db+dd]
            double rij2_scalar;
            _mm_store_sd(&rij2_scalar, total); // [da+dc+db+dd] 
            
            // Esto vale mas la pena con precision-simple
            //__m256d sumpd = _mm256_hadd_pd(rij2,rij2);
            //double rij2_scalar = _mm_cvtsd_f64(_mm_add_pd(_mm256_extractf128_pd(sumpd,0), _mm256_extractf128_pd(sumpd,1)));

            // Mask if rij2 <= rcut2
            if (rij2_scalar <= rcut2) {
                double r2inv = 1.0 / rij2_scalar;
                double r6inv = r2inv * r2inv * r2inv;

                double fr = 24.0 * r2inv * r6inv * (2.0 * r6inv - 1.0);

                __m256d frc = _mm256_mul_pd(_mm256_set1_pd(fr), rij);

                __m256d fi = _mm256_loadu_pd(fxyz + i);
                __m256d fj = _mm256_loadu_pd(fxyz + j);

                fi = _mm256_add_pd(fi, frc);
                fj = _mm256_sub_pd(fj, frc);

                _mm256_storeu_pd(fxyz + i, fi);
                _mm256_storeu_pd(fxyz + j, fj);

                *epot += 4.0 * r6inv * (r6inv - 1.0) - ECUT;
                pres_vir += fr * rij2_scalar;
            }
        }
    }

    pres_vir /= (V * 3.0);
    *pres = *temp * rho + pres_vir;
}


static inline double pbc(double cordi, const double cell_length)
{
    // condiciones periodicas de contorno coordenadas entre [0,L)
    if (cordi <= 0) {
        cordi += cell_length;
    } else if (cordi > cell_length) {
        cordi -= cell_length;
    }
    return cordi;
}


void velocity_verlet(double* restrict rxyz, double* restrict vxyz, double* restrict fxyz, double* epot,
        double* ekin, double* pres, double* temp, const double rho,
        const double V, const double L)
{

    for (int i = 0; i < 4 * N; i += 4) { // actualizo posiciones
        rxyz[i + 0] += vxyz[i + 0] * DT + 0.5 * fxyz[i + 0] * DT * DT;
        rxyz[i + 1] += vxyz[i + 1] * DT + 0.5 * fxyz[i + 1] * DT * DT;
        rxyz[i + 2] += vxyz[i + 2] * DT + 0.5 * fxyz[i + 2] * DT * DT;
        rxyz[i + 3] += vxyz[i + 3] * DT + 0.5 * fxyz[i + 3] * DT * DT;

        // TODO: Se me ocurre que esta bloque se puede sacar, porque es el unico que no podemos
        // vectorizar facilmente, debido a la llamada a pbc, que si bien esta inline capaz produce
        // que se desvien algunas lanes que van en paralelo debido al branching.
        rxyz[i + 0] = pbc(rxyz[i + 0], L);
        rxyz[i + 1] = pbc(rxyz[i + 1], L);
        rxyz[i + 2] = pbc(rxyz[i + 2], L);
        rxyz[i + 3] = pbc(rxyz[i + 3], L);

        vxyz[i + 0] += 0.5 * fxyz[i + 0] * DT;
        vxyz[i + 1] += 0.5 * fxyz[i + 1] * DT;
        vxyz[i + 2] += 0.5 * fxyz[i + 2] * DT;
        vxyz[i + 3] += 0.5 * fxyz[i + 3] * DT;
    }

    forces(rxyz, fxyz, epot, pres, temp, rho, V, L); // actualizo fuerzas

    double sumv2 = 0.0;
    for (int i = 0; i < 4 * N; i += 4) { // actualizo velocidades
        vxyz[i + 0] += 0.5 * fxyz[i + 0] * DT;
        vxyz[i + 1] += 0.5 * fxyz[i + 1] * DT;
        vxyz[i + 2] += 0.5 * fxyz[i + 2] * DT;
        vxyz[i + 3] += 0.5 * fxyz[i + 3] * DT;
    }

    for (int i = 0; i < 4 * N; i += 4) { // actualizo velocidades
        sumv2 += vxyz[i + 0] * vxyz[i + 0] + vxyz[i + 1] * vxyz[i + 1]
            + vxyz[i + 2] * vxyz[i + 2] + vxyz[i + 3] * vxyz[i + 3];
    }

    *ekin = 0.5 * sumv2;
    *temp = sumv2 / (3.0 * N);
}

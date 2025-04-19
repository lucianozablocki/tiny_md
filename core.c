#include "core.h"
#include "parameters.h"

#include <math.h>
#include <stdlib.h> // rand()

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


void forces(const double* rxyz, double* fxyz, double* epot, double* pres,
            const double* temp, const double rho, const double V, const double L)
{
    // calcula las fuerzas LJ (12-6)

    for (int i = 0; i < 4 * N; i++) {
        fxyz[i] = 0.0;
    }
    double pres_vir = 0.0;
    double rcut2 = RCUT * RCUT;
    *epot = 0.0;

    for (int i = 0; i < 4 * (N - 1); i += 4) {

        double xi = rxyz[i + 0];
        double yi = rxyz[i + 1];
        double zi = rxyz[i + 2];
        double wi = rxyz[i + 3];

        for (int j = i + 4; j < 4 * N; j += 4) {

            double xj = rxyz[j + 0];
            double yj = rxyz[j + 1];
            double zj = rxyz[j + 2];
            double wj = rxyz[j + 3];

            // distancia mínima entre r_i y r_j
            double rx = xi - xj;
            double ry = yi - yj;
            double rz = zi - zj;
            double rw = wi - wj;
            rx = minimum_image(rx, L);
            ry = minimum_image(ry, L);
            rz = minimum_image(rz, L);

            double rij2 = rx * rx + ry * ry + rz * rz + rw * rw;

            if (rij2 <= rcut2) {
                double r2inv = 1.0 / rij2;
                double r6inv = r2inv * r2inv * r2inv;

                double fr = 24.0 * r2inv * r6inv * (2.0 * r6inv - 1.0);

                fxyz[i + 0] += fr * rx;
                fxyz[i + 1] += fr * ry;
                fxyz[i + 2] += fr * rz;
                fxyz[i + 3] += fr * rw;

                fxyz[j + 0] -= fr * rx;
                fxyz[j + 1] -= fr * ry;
                fxyz[j + 2] -= fr * rz;
                fxyz[j + 3] -= fr * rw;

                *epot += 4.0 * r6inv * (r6inv - 1.0) - ECUT;
                pres_vir += fr * rij2;
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

#define _XOPEN_SOURCE 500  // M_PI
#include "core.h"
#include "parameters.h"
#include "wtime.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mtwister.h"

int main()
{
    FILE *file_xyz, *file_thermo;
    file_xyz = fopen("trajectory.xyz", "w");
    file_thermo = fopen("thermo.log", "w");
    float Ekin, Epot, Temp, Pres; // variables macroscopicas
    float Rho, cell_V, cell_L, tail, Etail, Ptail;
    float *rxyz, *vxyz, *fxyz; // variables microscopicas

    rxyz = (float*)calloc(4 * (N+2), sizeof(float));
    vxyz = (float*)calloc(4 * (N+2), sizeof(float));
    fxyz = (float*)calloc(4 * (N+2), sizeof(float));

    printf("# N\u00famero de part\u00edculas:      %d\n", N);
    printf("# Temperatura de referencia: %.2f\n", T0);
    printf("# Pasos de equilibraci\u00f3n:    %d\n", TEQ);
    printf("# Pasos de medici\u00f3n:         %d\n", TRUN - TEQ);
    printf("# (mediciones cada %d pasos)\n", TMES);
    printf("# densidad, volumen, energ\u00eda potencial media, presi\u00f3n media\n");
    fprintf(file_thermo, "# t Temp Pres Epot Etot\n");

    double t = 0.0;
    float sf;
    float Rhob;
    Rho = RHOI;
    init_pos(rxyz, Rho);
    double start = wtime();
    MTRand r = seedRand(SEED);
    for (int m = 0; m < 9; m++) {
        Rhob = Rho;
        Rho = RHOI - 0.1f * (float)m;
        cell_V = (float)N / Rho;
        cell_L = cbrtf(cell_V);
        tail = 16.0f * (float)M_PI * Rho * ((2.0f / 3.0f) * powf(RCUT, -9.0f) - powf(RCUT, -3.0f)) / 3.0f;
        Etail = tail * (float)N;
        Ptail = tail * Rho;

        int i = 0;
        sf = cbrtf(Rhob / Rho);
        for (int k = 0; k < 4 * N; k++) { // reescaleo posiciones a nueva densidad
            rxyz[k] *= sf;
        }
        init_vel(vxyz, &Temp, &Ekin, &r);
        forces(rxyz, fxyz, &Epot, &Pres, &Temp, Rho, cell_V, cell_L);

        for (i = 1; i < TEQ; i++) { // loop de equilibracion

            velocity_verlet(rxyz, vxyz, fxyz, &Epot, &Ekin, &Pres, &Temp, Rho, cell_V, cell_L);

            sf = sqrtf(T0 / Temp);
            for (int k = 0; k < 4 * N; k++) { // reescaleo de velocidades
                vxyz[k] *= sf;
            }
        }

        int mes = 0;
        float epotm = 0.0f, presm = 0.0f;
        for (i = TEQ; i < TRUN; i++) { // loop de medicion

            velocity_verlet(rxyz, vxyz, fxyz, &Epot, &Ekin, &Pres, &Temp, Rho, cell_V, cell_L);

            sf = sqrtf(T0 / Temp);
            for (int k = 0; k < 4 * N; k++) { // reescaleo de velocidades
                vxyz[k] *= sf;
            }

            if (i % TMES == 0) {
                Epot += Etail;
                Pres += Ptail;

                epotm += Epot;
                presm += Pres;
                mes++;

                fprintf(file_thermo, "%f %f %f %f %f\n", t, Temp, Pres, Epot, Epot + Ekin);
                fprintf(file_xyz, "%d\n\n", N);
                for (int k = 0; k < 4 * N; k += 4) {
                    fprintf(file_xyz, "Ar %e %e %e\n", rxyz[k + 0], rxyz[k + 1], rxyz[k + 2]);
                }
            }

            t += DT;
        }
        printf("%f\t%f\t%f\t%f\n", Rho, cell_V, epotm / (float)mes, presm / (float)mes);
    }

    double elapsed = wtime() - start;
    printf("# Tiempo total de simulaci\u00f3n = %f segundos\n", elapsed);
    printf("# Tiempo simulado = %f [fs]\n", t * 1.6);
    printf("# ns/day = %f\n", (1.6e-6 * t) / elapsed * 86400);
    //                       ^1.6 fs -> ns       ^sec -> day
    printf("*** particulas/s: %lf", N / elapsed);

    // Cierre de archivos
    fclose(file_thermo);
    fclose(file_xyz);

    // Liberacion de memoria
    free(rxyz);
    free(fxyz);
    free(vxyz);
    return 0;
}


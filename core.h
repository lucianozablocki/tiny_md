#ifndef CORE_H
#define CORE_H
#include <stdio.h>
#include "mtwister.h"

void init_pos(float* rxyz, const float rho);
void init_vel(float* vxyz, float* temp, float* ekin, MTRand* r);

#ifdef __cplusplus
extern "C" {
#endif

void forces(const float* rxyz, float* fxyz, float* epot, float* pres,
            const float* temp, float rho, float V, float L);

#ifdef __cplusplus
}
#endif

void velocity_verlet(float* rxyz, float* vxyz, float* fxyz, float* epot,
                     float* ekin, float* pres, float* temp, const float rho,
                     const float V, const float L);

#endif

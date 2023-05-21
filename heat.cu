#include <chrono>
#include <stdio.h>
#include <iostream>

#include "include/heatmap.h"
#include "include/lodepng.h"
#include "include/visualise.h"
#include "include/heatkernel.h"
#include "include/solve.h"
#include "include/solve_parallel.h"

int main()
{
    double dt = 0.0005;
    double dx = 0.1;
    double dy = 0.1;

    int nx = 2000;
    int ny = 2000;
    double t = 10;
    size_t timesteps = t / dt;

    double* boundary = (double*) calloc(nx * ny, sizeof(double));
    double* result = (double*) calloc(nx * ny, sizeof(double));

    heat_kernel(boundary, nx, ny);
    //heat_disc(boundary, nx, ny);

    auto start = std::chrono::high_resolution_clock::now();
    // solve heat equation
    //SolvePDE_explicitEuler(boundary, result, nx, ny, dx, dy, dt, timesteps);
    //SolvePDE_implicitEuler_Jacobian(boundary, result, nx, ny, dx, dy, dt, timesteps);
    //SolvePDE_explicitEuler_parallel(boundary, result, nx, ny, dx, dy, dt, timesteps);
    SolvePDE_implicitEuler_Jacobian_parallel(boundary, result, nx, ny, dx, dy, dt, timesteps);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("Elapsed time: %f\n", elapsed.count());
    // visualise result
    visualise(result, nx, ny);
}
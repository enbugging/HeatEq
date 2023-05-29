#include <chrono>
#include <stdio.h>
#include <iostream>

#include "include/heatmap.h"
#include "include/lodepng.h"
#include "include/visualise.h"
#include "include/heatkernel.h"
#include "include/solve.h"
#include "include/solve_parallel.h"
#include "include/solve_parallel_shared_memory.h"

int main(int argc, char** argv)
{
    double dt = 0.0005;
    double dx = 0.1;
    double dy = 0.1;

    int nx = 2000;
    int ny = 2000;
    double t = 1;
    int mode = 0;
    size_t timesteps = t / dt;

    switch(argc)
    {
        case 1:
            break;
        case 2:
            mode = atoi(argv[1]);
            break;
        case 3:
            mode = atoi(argv[1]);
            nx = atoi(argv[2]);
            ny = atoi(argv[2]);
            break;
        case 4:
            mode = atoi(argv[1]);
            nx = atoi(argv[2]);
            ny = atoi(argv[3]);
            break;
        case 5:
            mode = atoi(argv[1]);
            nx = atoi(argv[2]);
            ny = atoi(argv[3]);
            t = atoi(argv[4]);
            timesteps = t / dt;
            break;
        case 6:
            mode = atoi(argv[1]);
            nx = atoi(argv[2]);
            ny = atoi(argv[3]);
            t = atoi(argv[4]);
            timesteps = t / dt;
            if(atoi(argv[5]) == 1)
                heat_disc(boundary, nx, ny);
            else
                heat_kernel(boundary, nx, ny);
            break;
        default:
            printf("Usage: ./heat [mode] [nx] [ny] [t] [initial]\n");
            return 0;
    }

    double* boundary = (double*) calloc(nx * ny, sizeof(double));
    double* result = (double*) calloc(nx * ny, sizeof(double));

    auto start = std::chrono::high_resolution_clock::now();

    // solve heat equation
    switch (mode)
    {
        case 0:
            printf("Solving PDE with explicit Euler, sequential version\n");
            SolvePDE_explicitEuler(boundary, result, nx, ny, dx, dy, dt, timesteps);
            break;
        case 1:
            printf("Solving PDE with implicit Euler, sequential version\n");
            SolvePDE_implicitEuler_Jacobian(boundary, result, nx, ny, dx, dy, dt, timesteps);
            break;
        case 2:
            printf("Solving PDE with explicit Euler, parallel version\n");
            SolvePDE_explicitEuler_parallel(boundary, result, nx, ny, dx, dy, dt, timesteps);
            break;
        case 3:
            printf("Solving PDE with implicit Euler, parallel version\n");
            SolvePDE_implicitEuler_Jacobian_parallel(boundary, result, nx, ny, dx, dy, dt, timesteps);
            break;
        case 4:
            printf("Solving PDE with explicit Euler, parallel version with shared memory\n");
            SolvePDE_explicitEuler_parallel_shared_memory(boundary, result, nx, ny, dx, dy, dt, timesteps);
            break;
        case 5:
            printf("Solving PDE with implicit Euler, parallel version with shared memory\n");
            SolvePDE_implicitEuler_Jacobian_parallel_shared_memory(boundary, result, nx, ny, dx, dy, dt, timesteps);
            break;
        default:
            printf("Invalid mode\n");
            return 0;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("Elapsed time: %f\n", elapsed.count());
    // visualise result
    visualise(result, nx, ny);
}
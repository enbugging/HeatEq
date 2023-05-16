void SolvePDE_explicitEuler(
    double* boundary, 
    double* result, 
    size_t Nx,
    size_t Ny, 
    double dx, 
    double dy, 
    double dt,
    size_t timesteps,  
    double C = 1)
{
    double* curr = (double*) malloc(Nx * Ny * sizeof(double));
    double* next = (double*) malloc(Nx * Ny * sizeof(double));
    memcpy(curr, boundary, Nx * Ny * sizeof(double));
    // boundary condition
    for (size_t i = 0; i < Nx; i++) next[i * Ny] = curr[i * Ny], next[i * Ny + Ny - 1] = curr[i * Ny + Ny - 1];
    for (size_t j = 0; j < Ny; j++) next[j] = curr[j], next[(Nx - 1) * Ny + j] = curr[(Nx - 1) * Ny + j];
    for (size_t iter = 0; iter < timesteps; iter++)
    {
        for (size_t i = 1; i < Nx-1; i++)
        {
            for (size_t j = 1; j < Ny-1; j++)
            {
                next[i * Ny + j] = curr[i * Ny + j] + C * dt * (
                    (curr[(i + 1) * Ny + j] - 2 * curr[i * Ny + j] + curr[(i - 1) * Ny + j]) / (dx * dx) +
                    (curr[i * Ny + j + 1] - 2 * curr[i * Ny + j] + curr[i * Ny + j - 1]) / (dy * dy)
                );
            }
        }
        std::swap(curr, next);
    }
    memcpy(result, curr, Nx * Ny * sizeof(double));
    free(curr);
    free(next);
}

void SolvePDE_implicitEuler_Jacobian(
    double* boundary, 
    double* result, 
    size_t Nx,
    size_t Ny, 
    double dx, 
    double dy, 
    double dt,
    size_t timesteps,  
    double C = 1)
{
    double inv_a_ii = 1./(1 + 2*C*dt*(dx*dx + dy*dy)/(dx*dx*dy*dy));
    double* curr = (double*) malloc(Nx * Ny * sizeof(double));
    double* next = (double*) malloc(Nx * Ny * sizeof(double));
    double* next_prime = (double*) malloc(Nx * Ny * sizeof(double));
    memcpy(curr, boundary, Nx * Ny * sizeof(double));
    memcpy(next, boundary, Nx * Ny * sizeof(double));
    
    // boundary condition
    for (size_t i = 0; i < Nx; i++)
    {
        next_prime[i * Ny] = curr[i * Ny];
        next_prime[i * Ny + Ny - 1] = curr[i * Ny + Ny - 1];
    }
    for (size_t j = 0; j < Ny; j++)
    {
        next_prime[j] = curr[j];
        next_prime[(Nx - 1) * Ny + j] = curr[(Nx - 1) * Ny + j];
    }
    
    const size_t MAX_JAC_ITER = 100;
    double eps = 1e-6;

    for (size_t iter = 0; iter < timesteps; iter++)
    {
        for (size_t jac_iter = 0; jac_iter < MAX_JAC_ITER; jac_iter++)
        {
            double delta = 0;
            for (size_t i = 1; i < Nx-1; i++)
            {
                for (size_t j = 1; j < Ny-1; j++)
                {
                    next_prime[i * Ny + j] = inv_a_ii * (
                        curr[i * Ny + j] + C * dt * (
                            (next[(i + 1) * Ny + j] + next[(i - 1) * Ny + j]) / (dx * dx) +
                            (next[i * Ny + j + 1] + next[i * Ny + j - 1]) / (dy * dy)));
                    delta = std::max(delta, std::abs(next_prime[i * Ny + j] - next[i * Ny + j]));
                }
            }
            std::swap(next, next_prime);
            if (delta < eps) break;
        }
        std::swap(curr, next);
    }
    memcpy(result, curr, Nx * Ny * sizeof(double));
    free(curr);
    free(next);
}
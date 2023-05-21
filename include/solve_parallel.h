__global__
void SolvePDE_explicitEuler_parallel_aux(
    double* curr, 
    double* next, 
    size_t Nx, 
    size_t Ny, 
    double dx, 
    double dy, 
    double dt, 
    double C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= Nx || j >= Ny) return;
    if (i == 0 || i == Nx - 1 || j == 0 || j == Ny - 1) {
        next[i * Ny + j] = curr[i * Ny + j];
        return;
    }

    next[i * Ny + j] = curr[i * Ny + j] + C * dt * (
        (curr[(i + 1) * Ny + j] - 2 * curr[i * Ny + j] + curr[(i - 1) * Ny + j]) / (dx * dx) +
        (curr[i * Ny + j + 1] - 2 * curr[i * Ny + j] + curr[i * Ny + j - 1]) / (dy * dy)
    );
}

void SolvePDE_explicitEuler_parallel(
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
    // parallelization parameters
    size_t THREADS_PER_BLOCK = 16;
    dim3 blocks(Nx/THREADS_PER_BLOCK + 1, Ny/THREADS_PER_BLOCK + 1);
    dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    // moving the data to device
    size_t N = Nx * Ny;
    double* curr;
    double* next;
    cudaMalloc(&curr, N * sizeof(double));
    cudaMalloc(&next, N * sizeof(double));
    cudaMemcpy(curr, boundary, N * sizeof(double), cudaMemcpyHostToDevice);

    // computing on GPU
    for (size_t i = 0; i < timesteps; i++) {
        SolvePDE_explicitEuler_parallel_aux<<<blocks, threads>>>(curr, next, Nx, Ny, dx, dy, dt, C);
        cudaDeviceSynchronize();
        std::swap(curr, next);
    }

    // copying the result back
    cudaMemcpy(result, curr, N * sizeof(double), cudaMemcpyDeviceToHost);
  
    // Free memory
    cudaFree(curr);
    cudaFree(next);
}

__global__
void SolvePDE_implicitEuler_Jacobian_parallel_aux(
    double* curr, 
    double* next, 
    double* next_prime, 
    size_t Nx, 
    size_t Ny, 
    double dx, 
    double dy, 
    double dt, 
    double C,
    double inv_a_ii, 
    double eps, 
    bool* keep_going)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= Nx || j >= Ny) return;
    if (i == 0 || i == Nx - 1 || j == 0 || j == Ny - 1) {
        next_prime[i * Ny + j] = next[i * Ny + j];
        return;
    }

    next_prime[i * Ny + j] = 
        inv_a_ii * (
            curr[i * Ny + j] + 
            C * dt * (
                (next[(i + 1) * Ny + j] + next[(i - 1) * Ny + j]) / (dx * dx) +
                (next[i * Ny + j + 1] + next[i * Ny + j - 1]) / (dy * dy)
            )
        );
    if (std::abs(next_prime[i * Ny + j] - next[i * Ny + j]) >= eps) *keep_going = true;
}

void SolvePDE_implicitEuler_Jacobian_parallel(
    double* boundary, 
    double* result, 
    size_t Nx,
    size_t Ny, 
    double dx, 
    double dy, 
    double dt,
    size_t timesteps,  
    double C = 1, 
    size_t MAX_JAC_ITER = 100, 
    double eps = 1e-6)
{
    // parallelization parameters
    size_t THREADS_PER_BLOCK = 16;
    dim3 blocks(Nx/THREADS_PER_BLOCK + 1, Ny/THREADS_PER_BLOCK + 1);
    dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    // moving the data to device
    double inv_a_ii = 1./(1 + 2*C*dt*(dx*dx + dy*dy)/(dx*dx*dy*dy));
    size_t N = Nx * Ny;
    double* curr;
    double* next;
    double* next_prime;
    cudaMalloc(&curr, N * sizeof(double));
    cudaMalloc(&next, N * sizeof(double));
    cudaMalloc(&next_prime, N * sizeof(double));
    cudaMemcpy(curr, boundary, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(next, boundary, N * sizeof(double), cudaMemcpyHostToDevice);
    bool* keep_going;
    bool host_keep_going;
    cudaMalloc((void**)&keep_going, sizeof(bool));

    // computing on GPU
    for (size_t i = 0; i < timesteps; i++) {
        size_t max_iter = 0;
        for (size_t jac_iter = 0; jac_iter < MAX_JAC_ITER; jac_iter++)
        {
            host_keep_going = false;
            cudaMemcpy(keep_going, &host_keep_going, sizeof(bool), cudaMemcpyHostToDevice);

            SolvePDE_implicitEuler_Jacobian_parallel_aux<<<blocks, threads>>>
                (curr, next, next_prime, Nx, Ny, dx, dy, dt, C, inv_a_ii, eps, keep_going);
            cudaDeviceSynchronize();
            
            std::swap(next, next_prime);
            max_iter = std::max(max_iter, jac_iter);
            cudaMemcpy(&host_keep_going, keep_going, sizeof(bool), cudaMemcpyDeviceToHost);
            if (!host_keep_going) break;
        }
        std::swap(curr, next);
    }

    // copying the result back
    cudaMemcpy(result, curr, N * sizeof(double), cudaMemcpyDeviceToHost);
  
    // Free memory
    cudaFree(curr);
    cudaFree(next);
    cudaFree(next_prime);
    cudaFree(keep_going);
}
#ifndef HEATKERNEL_H
#define HEATKERNEL_H

void heat_kernel(
    double* boundary, 
    int nx, 
    int ny)
{
    boundary[nx/2 * ny + ny/2] = 10;
}

void heat_disc(
    double* boundary, 
    int nx, 
    int ny)
{
    double radius1 = (nx/5) * (ny/5);
    double radius2 = (nx/4) * (ny/4);
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            if ((i - nx/2) * (i - nx/2) + (j - ny/2)*(j - ny/2) < radius2 &&
                (i - nx/2) * (i - nx/2) + (j - ny/2)*(j - ny/2) > radius1)
            {
                boundary[i * ny + j] = 10;
            }
        }
    }
}

#endif
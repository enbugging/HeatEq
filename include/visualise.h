void visualise(
    double* result,
    int nx,
    int ny)
{
    heatmap_t* hm = heatmap_new(nx, ny);
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            heatmap_add_weighted_point(hm, i, j, result[i * ny + j]);
        }
    }

    // This creates an image out of the heatmap.
    // `image` now contains the image data in 32-bit RGBA.
    unsigned char* image = (unsigned char*) malloc(nx * ny * 4 * sizeof(unsigned char));
    heatmap_render_default_to(hm, &image[0]);

    // Now that we've got a finished heatmap picture, we don't need the map anymore.
    heatmap_free(hm);

    // Finally, we use the fantastic lodepng library to save it as an image.
    if(unsigned error = lodepng::encode("heatmap.png", image, nx, ny)) {
        std::cerr << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
    }
    free(image);
}
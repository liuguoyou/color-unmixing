#include "image_processing.h"

namespace ImageProcessing
{

template <typename Scalar>
Scalar crop(Scalar x, Scalar min_x, Scalar max_x)
{
    return std::max(std::min(x, max_x), min_x);
}

Image apply_convolution(const Image& image, const Eigen::MatrixXd &kernel)
{
    const int w = image.width();
    const int h = image.height();

    const int kernel_size = kernel.rows();

    assert(kernel_size % 2 == 1);
    assert(kernel_size == kernel.cols());

    Image new_image(w, h);

    for (int x = 0; x < w; ++ x)
    {
        for (int y = 0; y < h; ++ y)
        {
            Image::Value value = 0.0;
            for (int kernel_x = 0; kernel_x < kernel_size; ++ kernel_x)
            {
                for (int kernel_y = 0; kernel_y < kernel_size; ++ kernel_y)
                {
                    int original_image_x = crop(x + kernel_x - ((kernel_size - 1) / 2), 0, w - 1);
                    int original_image_y = crop(y + kernel_y - ((kernel_size - 1) / 2), 0, h - 1);

                    value += kernel(kernel_x, kernel_y) * image.get_pixel(original_image_x, original_image_y);
                }
            }
            new_image.set_pixel(x, y, value);
        }
    }

    return new_image;
}

}

#include "image_processing.h"
#include <numeric>
#include <QImage>

namespace ImageProcessing
{

void Image::normalize()
{
    const double sum = std::accumulate(pixels.begin(), pixels.end(), 0.0);
    assert(sum > 1e-16);
    for (int x = 0; x < width(); ++ x)
    {
        for (int y = 0; y < height(); ++ y)
        {
            set_pixel(x, y, get_pixel(x, y) / sum);
        }
    }
}

void Image::save(const std::string &file_path) const
{
    QImage q_image(width(), height(), QImage::Format_ARGB32);
    for (int x = 0; x < width(); ++ x)
    {
        for (int y = 0; y < height(); ++ y)
        {
            const double value = get_pixel(x, y);
            q_image.setPixelColor(x, y, QColor(value * 255, value * 255, value * 255, 255));
        }
    }
    q_image.save(QString::fromStdString(file_path));
}

ColorImage::ColorImage(const std::string &file_path)
{
    QImage q_image(QString::fromStdString(file_path));
    width_  = q_image.width();
    height_ = q_image.height();

    assert(width() > 0 && height() > 0);

    rgba_ = std::vector<Image>(4, Image(width(), height()));
    for (int x = 0; x < width(); ++ x)
    {
        for (int y = 0; y < height(); ++ y)
        {
            const QColor q_color = q_image.pixelColor(x, y);
            rgba_[0].set_pixel(x, y, q_color.redF());
            rgba_[1].set_pixel(x, y, q_color.greenF());
            rgba_[2].set_pixel(x, y, q_color.blueF());
            rgba_[3].set_pixel(x, y, q_color.alphaF());
        }
    }
}

Image ColorImage::get_luminance() const
{
    Image new_image(width(), height());
    for (int x = 0; x < width(); ++ x)
    {
        for (int y = 0; y < height(); ++ y)
        {
            const double r = rgba_[0].get_pixel(x, y);
            const double g = rgba_[1].get_pixel(x, y);
            const double b = rgba_[2].get_pixel(x, y);

            // https://en.wikipedia.org/wiki/Relative_luminance
            const double luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;

            new_image.set_pixel(x, y, luminance);
        }
    }
    return new_image;
}

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
            double value = 0.0;
            for (int kernel_x = 0; kernel_x < kernel_size; ++ kernel_x)
            {
                for (int kernel_y = 0; kernel_y < kernel_size; ++ kernel_y)
                {
                    const int original_image_x = crop(x + kernel_x - ((kernel_size - 1) / 2), 0, w - 1);
                    const int original_image_y = crop(y + kernel_y - ((kernel_size - 1) / 2), 0, h - 1);

                    value += kernel(kernel_x, kernel_y) * image.get_pixel(original_image_x, original_image_y);
                }
            }
            new_image.set_pixel(x, y, value);
        }
    }

    return new_image;
}

}

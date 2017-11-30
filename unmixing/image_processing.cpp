#include "image_processing.h"
#include <numeric>
#include <cfloat>
#include <Eigen/LU>
#include <QImage>

namespace ImageProcessing
{

template <typename Scalar>
Scalar crop(Scalar x, Scalar min_x, Scalar max_x)
{
    return std::max(std::min(x, max_x), min_x);
}

void Image::force_unity()
{
    const double sum = std::accumulate(pixels.begin(), pixels.end(), 0.0);
    assert(sum > 1e-16);
    for (int x = 0; x < width(); ++ x) for (int y = 0; y < height(); ++ y)
    {
        set_pixel(x, y, get_pixel(x, y) / sum);
    }
}

void Image::scale_to_unit()
{
    double max_value = - DBL_MAX;
    double min_value = + DBL_MAX;

    for (int x = 0; x < width(); ++ x) for (int y = 0; y < height(); ++ y)
    {
        const double value = get_pixel(x, y);
        max_value = std::max(max_value, value);
        min_value = std::min(min_value, value);
    }
    assert(max_value - min_value > 0.0);
    for (int x = 0; x < width(); ++ x) for (int y = 0; y < height(); ++ y)
    {
        set_pixel(x, y, (get_pixel(x, y) - min_value) / (max_value - min_value));
    }
}

void AbstractImage::save(const std::string &file_path) const
{
    QImage q_image(width(), height(), QImage::Format_ARGB32);
    for (int x = 0; x < width(); ++ x) for (int y = 0; y < height(); ++ y)
    {
        const IntColor color = get_color(x, y);
        q_image.setPixelColor(x, y, QColor(color(0), color(1), color(2), color(3)));
    }
    q_image.save(QString::fromStdString(file_path));
}

AbstractImage::IntColor Image::get_color(int x, int y) const
{
    const double value = crop(get_pixel(x, y), 0.0, 1.0);
    return IntColor(value * 255, value * 255, value * 255, 255);
}

AbstractImage::IntColor ColorImage::get_color(int x, int y) const
{
    Eigen::Vector4d color = get_rgba(x, y);
    for (int i : { 0, 1, 2, 3 }) color(i) = crop(color(i), 0.0, 1.0);
    return IntColor(color(0) * 255, color(1) * 255, color(2) * 255, color(3) * 255);
}

ColorImage::ColorImage(const std::string &file_path)
{
    QImage q_image(QString::fromStdString(file_path));
    width_  = q_image.width();
    height_ = q_image.height();

    assert(width() > 0 && height() > 0);

    rgba_ = std::vector<Image>(4, Image(width(), height()));
    for (int x = 0; x < width(); ++ x) for (int y = 0; y < height(); ++ y)
    {
        const QColor q_color = q_image.pixelColor(x, y);
        rgba_[0].set_pixel(x, y, q_color.redF());
        rgba_[1].set_pixel(x, y, q_color.greenF());
        rgba_[2].set_pixel(x, y, q_color.blueF());
        rgba_[3].set_pixel(x, y, q_color.alphaF());
    }
}

Image ColorImage::get_luminance() const
{
    Image new_image(width(), height());
    for (int x = 0; x < width(); ++ x) for (int y = 0; y < height(); ++ y)
    {
        const double r = rgba_[0].get_pixel(x, y);
        const double g = rgba_[1].get_pixel(x, y);
        const double b = rgba_[2].get_pixel(x, y);

        // https://en.wikipedia.org/wiki/Relative_luminance
        const double luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;

        new_image.set_pixel(x, y, luminance);
    }
    return new_image;
}

///////////////////////////////////////////////////////////////////////////////////////

Image apply_convolution(const Image& image, const Eigen::MatrixXd &kernel)
{
    const int w = image.width();
    const int h = image.height();

    const int kernel_size = kernel.rows();

    assert(kernel_size % 2 == 1);
    assert(kernel_size == kernel.cols());

    Image new_image(w, h);

    for (int x = 0; x < w; ++ x) for (int y = 0; y < h; ++ y)
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
    return new_image;
}

Image calculate_guided_filter_kernel(const Image& image, int center_x, int center_y, int radius, double epsilon, bool force_positive)
{
    const int width  = image.width();
    const int height = image.height();

    const Image mean_I = ImageProcessing::apply_box_filter(image, radius);
    const Image corr_I = ImageProcessing::apply_box_filter(ImageProcessing::square(image), radius);
    const Image var_I  = ImageProcessing::subtract(corr_I, ImageProcessing::square(mean_I));
    const double I_seed = image.get_pixel(center_x, center_y);

    Image weight_map(width, height, 0.0);
    for (int x = center_x - radius; x <= center_x + radius; ++ x)
    {
        for (int y = center_y - radius; y <= center_y + radius; ++ y)
        {
            if (x < 0 || x >= width || y < 0 || y >= height) continue;

            const double I_j = image.get_pixel(x, y);
            double weight = 0.0;
            for (int k_x = 0; k_x < width; ++ k_x)
            {
                for (int k_y = 0; k_y < height; ++ k_y)
                {
                    bool range_j    = (k_x >= x - radius && k_x <= x + radius) && (k_y >= y - radius && k_y <= y + radius);
                    bool range_seed = (k_x >= center_x - radius && k_x <= center_x + radius) && (k_y >= center_y - radius && k_y <= center_y + radius);
                    if (!range_j || !range_seed) continue;

                    const double mu_k  = mean_I.get_pixel(k_x, k_y);
                    const double var_k = var_I.get_pixel(k_x, k_y);
                    weight += 1.0 + ((I_seed - mu_k) * (I_j - mu_k)) / (epsilon + var_k);
                }
            }
            if (force_positive) weight = std::max(0.0, weight);
            weight_map.set_pixel(x, y, weight);
        }
    }
    weight_map.force_unity();
    return weight_map;
}

Image calculate_gradient_magnitude(const Image& image)
{
    const int width  = image.width();
    const int height = image.height();
    const Image sobel_x = ImageProcessing::apply_sobel_filter_x(image);
    const Image sobel_y = ImageProcessing::apply_sobel_filter_y(image);

    Image gradient_magnitude = Image(width, height);
    for (int x = 0; x < width; ++ x)
    {
        for (int y = 0; y < height; ++ y)
        {
            const double g_x = sobel_x.get_pixel(x, y);
            const double g_y = sobel_y.get_pixel(x, y);
            const double value = std::sqrt(g_x * g_x + g_y * g_y);
            gradient_magnitude.set_pixel(x, y, value);
        }
    }
    return gradient_magnitude;
}

Image apply_guided_filter(const Image& input_image, const ColorImage& guidance_image, int radius, double epsilon)
{
    const int width  = input_image.width();
    const int height = input_image.height();

    assert(width == guidance_image.width());
    assert(height == guidance_image.height());

    const Image I_r = guidance_image.get_r();
    const Image I_g = guidance_image.get_g();
    const Image I_b = guidance_image.get_b();

    const Image mean_I_r = apply_box_filter(I_r, radius);
    const Image mean_I_g = apply_box_filter(I_g, radius);
    const Image mean_I_b = apply_box_filter(I_b, radius);

    const Image mean_p = apply_box_filter(input_image, radius);

    const Image mean_Ip_r = apply_box_filter(multiply(I_r, input_image), radius);
    const Image mean_Ip_g = apply_box_filter(multiply(I_g, input_image), radius);
    const Image mean_Ip_b = apply_box_filter(multiply(I_b, input_image), radius);

    const Image cov_Ip_r = subtract(mean_Ip_r, multiply(mean_I_r, mean_p));
    const Image cov_Ip_g = subtract(mean_Ip_g, multiply(mean_I_g, mean_p));
    const Image cov_Ip_b = subtract(mean_Ip_b, multiply(mean_I_b, mean_p));

    const Image var_I_rr = subtract(apply_box_filter(multiply(I_r, I_r), radius), multiply(mean_I_r, mean_I_r));
    const Image var_I_rg = subtract(apply_box_filter(multiply(I_r, I_g), radius), multiply(mean_I_r, mean_I_g));
    const Image var_I_rb = subtract(apply_box_filter(multiply(I_r, I_b), radius), multiply(mean_I_r, mean_I_b));
    const Image var_I_gg = subtract(apply_box_filter(multiply(I_g, I_g), radius), multiply(mean_I_g, mean_I_g));
    const Image var_I_gb = subtract(apply_box_filter(multiply(I_g, I_b), radius), multiply(mean_I_g, mean_I_b));
    const Image var_I_bb = subtract(apply_box_filter(multiply(I_b, I_b), radius), multiply(mean_I_b, mean_I_b));

    Image a_r(width, height);
    Image a_g(width, height);
    Image a_b(width, height);
    for (int x = 0; x < width; ++ x) for (int y = 0; y < height; ++ y)
    {
        Eigen::Matrix3d sigma;
        sigma << var_I_rr.get_pixel(x, y), var_I_rg.get_pixel(x, y), var_I_rb.get_pixel(x, y),
                 var_I_rg.get_pixel(x, y), var_I_gg.get_pixel(x, y), var_I_gb.get_pixel(x, y),
                 var_I_rb.get_pixel(x, y), var_I_gb.get_pixel(x, y), var_I_bb.get_pixel(x, y);

        const Eigen::Vector3d cov_Ip = { cov_Ip_r.get_pixel(x, y), cov_Ip_g.get_pixel(x, y), cov_Ip_b.get_pixel(x, y) };
        const Eigen::Vector3d a_xy = (sigma + epsilon * Eigen::Matrix3d::Identity()).inverse() * cov_Ip;

        a_r.set_pixel(x, y, a_xy(0));
        a_g.set_pixel(x, y, a_xy(1));
        a_b.set_pixel(x, y, a_xy(2));
    }

    const Image b = subtract(subtract(subtract(mean_p, multiply(a_r, mean_I_r)), multiply(a_g, mean_I_g)), multiply(a_b, mean_I_g));

    Image q = apply_box_filter(b, radius);
    q = add(q, multiply(apply_box_filter(a_r, radius), I_r));
    q = add(q, multiply(apply_box_filter(a_g, radius), I_g));
    q = add(q, multiply(apply_box_filter(a_b, radius), I_b));

    return q;
}


}

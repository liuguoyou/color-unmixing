#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <vector>
#include <Eigen/Core>

namespace ImageProcessing
{

class Image
{
public:
    using Value = double;

    Image(int width, int height, const Value& value = 0.0) : width_(width), height_(height)
    {
        pixels = std::vector<Value>(width_ * height_, value);
    }

    void set_pixel(int x, int y, const Value& value)
    {
        assert(x < width() && y < height());
        pixels[y * width() + x] = value;
    }

    const Value& get_pixel(int x, int y) const
    {
        assert(x < width() && y < height());
        return pixels[y * width() + x];
    }

    void normalize();

    int width()  const { return width_; }
    int height() const { return height_; }

private:
    int width_;
    int height_;

    std::vector<Value> pixels;
};

Image apply_convolution(const Image& image, const Eigen::MatrixXd& kernel);

///////////////////////////////////////////////////////////////////////////////////
// Wrapper functions
///////////////////////////////////////////////////////////////////////////////////

inline Image apply_sobel_filter_x(const Image& image)
{
    Eigen::Matrix3d kernel;
    kernel << +1.0,  0.0, -1.0,
              +2.0,  0.0, -2.0,
              +1.0,  0.0, -1.0;
    return apply_convolution(image, kernel);
}

inline Image apply_sobel_filter_y(const Image& image)
{
    Eigen::Matrix3d kernel;
    kernel << +1.0, +2.0, +1.0,
               0.0,  0.0,  0.0,
              -1.0, -2.0, -1.0;
    return apply_convolution(image, kernel);
}

inline Image apply_box_filter(const Image& image, int radius)
{
    assert(radius >= 0);
    if (radius == 0) return image;
    const int size = 2 * radius + 1;
    const Eigen::MatrixXd kernel = Eigen::MatrixXd::Constant(size, size, 1.0 / static_cast<double>(size * size));
    return apply_convolution(image, kernel);
}

inline Image square(const Image& image)
{
    Image new_image(image.width(), image.height());
    for (int x = 0; x < image.width(); ++ x)
    {
        for (int y = 0; y < image.height(); ++ y)
        {
            new_image.set_pixel(x, y, image.get_pixel(x, y) * image.get_pixel(x, y));
        }
    }
    return new_image;
}

inline Image substitute(const Image& left_image, const Image& right_image)
{
    assert(left_image.width() == right_image.width());
    assert(left_image.height() == right_image.height());
    Image new_image(left_image.width(), left_image.height());
    for (int x = 0; x < left_image.width(); ++ x)
    {
        for (int y = 0; y < left_image.height(); ++ y)
        {
            new_image.set_pixel(x, y, left_image.get_pixel(x, y) - right_image.get_pixel(x, y));
        }
    }
    return new_image;
}

}

#endif // IMAGE_PROCESSING_H

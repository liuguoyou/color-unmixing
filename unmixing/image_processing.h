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

    Image(int width, int height) : width_(width), height_(height)
    {
        pixels = std::vector<Value>(width_ * height_);
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

}

#endif // IMAGE_PROCESSING_H

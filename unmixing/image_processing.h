#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <vector>
#include <string>
#include <Eigen/Core>

namespace ImageProcessing
{

class AbstractImage
{
public:
    AbstractImage(int width, int height) : width_(width), height_(height) {}
    AbstractImage() {}

    using IntColor = Eigen::Vector4i;

    void save(const std::string& file_path) const;

    int width()  const { return width_;  }
    int height() const { return height_; }

protected:
    int width_;
    int height_;

    virtual IntColor get_color(int x, int y) const = 0;
};

class Image : public AbstractImage
{
public:
    Image(int width, int height, double value = 0.0) : AbstractImage(width, height)
    {
        pixels = std::vector<double>(width_ * height_, value);
    }

    void set_pixel(int x, int y, double value)
    {
        assert(x < width() && y < height());
        pixels[y * width() + x] = value;
    }

    double get_pixel(int x, int y) const
    {
        assert(x < width() && y < height());
        return pixels[y * width() + x];
    }

    void force_unity();
    void scale_to_unit();

private:
    IntColor get_color(int x, int y) const;

    std::vector<double> pixels;
};

class ColorImage : public AbstractImage
{
public:
    ColorImage(int width, int height) : AbstractImage(width, height)
    {
        rgba_ = std::vector<Image>(4, Image(width, height));
    }

    ColorImage(const std::string& file_path);

    void set_rgb(int x, int y, const Eigen::Vector3d& rgb)
    {
        assert(x < width() && y < height());
        for (int i : { 0, 1, 2 }) rgba_[i].set_pixel(x, y, rgb(i));
    }

    void set_rgba(int x, int y, const Eigen::Vector4d& rgba)
    {
        assert(x < width() && y < height());
        for (int i : { 0, 1, 2, 3 }) rgba_[i].set_pixel(x, y, rgba(i));
    }

    void set_rgba(int x, int y, const Eigen::Vector3d& rgb, double a)
    {
        assert(x < width() && y < height());
        for (int i : { 0, 1, 2 }) rgba_[i].set_pixel(x, y, rgb(i));
        rgba_[3].set_pixel(x, y, a);
    }

    Eigen::Vector3d get_rgb(int x, int y) const
    {
        assert(x < width() && y < height());
        Eigen::Vector3d rgb;
        for (int i : { 0, 1, 2 }) rgb(i) = rgba_[i].get_pixel(x, y);
        return rgb;
    }

    Eigen::Vector4d get_rgba(int x, int y) const
    {
        assert(x < width() && y < height());
        Eigen::Vector4d rgba;
        for (int i : { 0, 1, 2, 3 }) rgba(i) = rgba_[i].get_pixel(x, y);
        return rgba;
    }

    Image get_luminance() const;

    Image& get_r() { return rgba_[0]; }
    Image& get_g() { return rgba_[1]; }
    Image& get_b() { return rgba_[2]; }
    Image& get_a() { return rgba_[3]; }
    const Image& get_r() const { return rgba_[0]; }
    const Image& get_g() const { return rgba_[1]; }
    const Image& get_b() const { return rgba_[2]; }
    const Image& get_a() const { return rgba_[3]; }
    void set_r(const Image& r) { rgba_[0] = r; }
    void set_g(const Image& g) { rgba_[1] = g; }
    void set_b(const Image& b) { rgba_[2] = b; }
    void set_a(const Image& a) { rgba_[3] = a; }

private: 
    IntColor get_color(int x, int y) const;

    std::vector<Image> rgba_;
};

Image apply_convolution(const Image& image, const Eigen::MatrixXd& kernel);
Image calculate_guided_filter_kernel(const Image& image, int center_x, int center_y, int radius, double epsilon = 0.01, bool force_positive = true);
Image calculate_gradient_magnitude(const Image& image);
Image apply_guided_filter(const Image& input_image, const ColorImage& guidance_image, int radius, double epsilon);

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

inline Image multiply(const Image& left_image, const Image& right_image)
{
    Image new_image(left_image.width(), left_image.height());
    for (int x = 0; x < left_image.width(); ++ x) for (int y = 0; y < left_image.height(); ++ y)
    {
        new_image.set_pixel(x, y, left_image.get_pixel(x, y) * right_image.get_pixel(x, y));
    }
    return new_image;
}

inline Image square(const Image& image)
{
    Image new_image(image.width(), image.height());
    for (int x = 0; x < image.width(); ++ x) for (int y = 0; y < image.height(); ++ y)
    {
        new_image.set_pixel(x, y, image.get_pixel(x, y) * image.get_pixel(x, y));
    }
    return new_image;
}

inline Image subtract(const Image& left_image, const Image& right_image)
{
    assert(left_image.width() == right_image.width());
    assert(left_image.height() == right_image.height());
    Image new_image(left_image.width(), left_image.height());
    for (int x = 0; x < left_image.width(); ++ x) for (int y = 0; y < left_image.height(); ++ y)
    {
        new_image.set_pixel(x, y, left_image.get_pixel(x, y) - right_image.get_pixel(x, y));
    }
    return new_image;
}

inline Image add(const Image& left_image, const Image& right_image)
{
    assert(left_image.width() == right_image.width());
    assert(left_image.height() == right_image.height());
    Image new_image(left_image.width(), left_image.height());
    for (int x = 0; x < left_image.width(); ++ x) for (int y = 0; y < left_image.height(); ++ y)
    {
        new_image.set_pixel(x, y, left_image.get_pixel(x, y) + right_image.get_pixel(x, y));
    }
    return new_image;
}

}

#endif // IMAGE_PROCESSING_H

#include "unmixing.h"
#include <cmath>
#include <cfloat>
#include <iostream>
#include <thread>
#include <QImage>
#include "nloptutility.h"

using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;
using Eigen::Matrix3d;

//#define SPARCITY
#define GRADIENT
//#define PARALLEL

namespace
{

#ifdef PARALLEL
// Perform the function in parallel for { 0, 1, ..., n - 1 }
template<typename Callable>
void perform_in_parallel(Callable function, int width, int height)
{
    const int hint      = std::thread::hardware_concurrency();
    const int n_threads = std::min(width * height, (hint == 0) ? 4 : hint);

    auto inner_loop = [width, height, n_threads, function](const int j)
    {
        const int n = width * height;

        const int start_index = j * (n / n_threads);
        const int end_index   = (j + 1 == n_threads) ? n : (j + 1) * (n / n_threads);

        for (int k = start_index; k < end_index; ++ k) function(k % width, k / width);
    };
    std::vector<std::thread> threads;
    for (int j = 0; j < n_threads; ++ j) threads.push_back(std::thread(inner_loop, j));
    for (auto& t : threads) t.join();
}
#endif

struct OptimizationParameterSet
{
    Vector3d target_color;
    Vector4d lambda;
    double   lo;
    double   sigma;
};

struct ColorKernel
{
    ColorKernel(const Vector3d& mu, const Matrix3d& sigma_inv) :
        mu(mu),
        sigma_inv(sigma_inv)
    {
    }

    Vector3d mu;
    Matrix3d sigma_inv;

    double calculate_squared_Mahalanobis_distance(const VectorXd& color) const
    {
        return (color - mu).transpose() * sigma_inv * (color - mu);
    }
};

std::vector<ColorKernel> kernels;

Vector3d composite_color(const VectorXd& alphas, const VectorXd& colors)
{
    const int number_of_layers = alphas.rows();
    Vector3d sum_color = Vector3d::Zero();
    for (int index = 0; index < number_of_layers; ++ index)
    {
        sum_color += alphas[index] * colors.segment<3>(index * 3);
    }
    return sum_color;
}

VectorXd gradient_of_equality_constraint_terms(const VectorXd& alphas, const VectorXd& colors, const Vector3d& target_color, const Vector4d& constraint_vector, const Vector4d& lambda, double lo)
{
    const int number_of_layers = alphas.rows();

    VectorXd grad_lagrange_term = VectorXd(number_of_layers * 4);
    VectorXd grad_penalty_term  = VectorXd(number_of_layers * 4);

    const double   sum_alpha = alphas.sum();
    const Vector3d sum_color = composite_color(alphas, colors);

    for (int index = 0; index < number_of_layers; ++ index)
    {
        const Vector3d& u = colors.segment<3>(index * 3);
        const double    a = alphas(index);

        const Vector3d partial_g_u_per_partial_a = 2.0 * u.cwiseProduct(sum_color - target_color);
        const double   partial_g_a_per_partial_a = 2.0 * (sum_alpha - 1.0);

        const double partial_lambda_transpose_g_per_partial_alpha = lambda.segment<3>(0).transpose() * partial_g_u_per_partial_a + lambda(3) * partial_g_a_per_partial_a;

        const double partial_g_u_r_per_partial_u_r = 2.0 * a * (sum_color(0) - target_color(0));
        const double partial_g_u_g_per_partial_u_g = 2.0 * a * (sum_color(1) - target_color(1));
        const double partial_g_u_b_per_partial_u_b = 2.0 * a * (sum_color(2) - target_color(2));

        const double partial_lambda_transpose_g_per_partial_u_r = lambda(0) * partial_g_u_r_per_partial_u_r;
        const double partial_lambda_transpose_g_per_partial_u_g = lambda(1) * partial_g_u_g_per_partial_u_g;
        const double partial_lambda_transpose_g_per_partial_u_b = lambda(2) * partial_g_u_b_per_partial_u_b;

        const Vector4d partial_g_per_partial_a = Vector4d(partial_g_u_per_partial_a(0), partial_g_u_per_partial_a(1), partial_g_u_per_partial_a(2), partial_g_a_per_partial_a);

        grad_lagrange_term(index)                            = partial_lambda_transpose_g_per_partial_alpha;
        grad_lagrange_term(number_of_layers + index * 3 + 0) = partial_lambda_transpose_g_per_partial_u_r;
        grad_lagrange_term(number_of_layers + index * 3 + 1) = partial_lambda_transpose_g_per_partial_u_g;
        grad_lagrange_term(number_of_layers + index * 3 + 2) = partial_lambda_transpose_g_per_partial_u_b;

        grad_penalty_term(index)                            = 0.5 * lo * 2.0 * constraint_vector.transpose() * partial_g_per_partial_a;
        grad_penalty_term(number_of_layers + index * 3 + 0) = 0.5 * lo * 2.0 * constraint_vector(0) * partial_g_u_r_per_partial_u_r;
        grad_penalty_term(number_of_layers + index * 3 + 1) = 0.5 * lo * 2.0 * constraint_vector(1) * partial_g_u_g_per_partial_u_g;
        grad_penalty_term(number_of_layers + index * 3 + 2) = 0.5 * lo * 2.0 * constraint_vector(2) * partial_g_u_b_per_partial_u_b;
    }

    return grad_lagrange_term + grad_penalty_term;
}

double calculate_equality_constraint_terms(const Vector4d& constraint_vector, const Vector4d& lambda, double lo)
{
    return lambda.transpose() * constraint_vector + 0.5 * lo * constraint_vector.squaredNorm();
}

Vector4d calculate_equality_constraint_vector(const VectorXd& alphas, const VectorXd& colors, const Vector3d& target_color)
{
    const double   sum_alpha = alphas.sum();
    const Vector3d sum_color = composite_color(alphas, colors);

    const double   g_alpha = (sum_alpha - 1.0) * (sum_alpha - 1.0);
    const Vector3d g_color = (sum_color - target_color).cwiseProduct(sum_color - target_color);

    return Vector4d(g_color(0), g_color(1), g_color(2), g_alpha);
}

Vector4d calculate_equality_constraint_vector(const VectorXd& x, const Vector3d& target_color)
{
    const int number_of_layers = x.rows() / 4;
    return calculate_equality_constraint_vector(x.segment(0, number_of_layers), x.segment(number_of_layers, number_of_layers * 3), target_color);
}

VectorXd gradient_of_energy_function(const VectorXd& alphas, const VectorXd& colors, double sigma)
{
    const int number_of_layers = alphas.rows();

    VectorXd grad = VectorXd(number_of_layers * 4);

    // Main term
    for (int index = 0; index < number_of_layers; ++ index)
    {
        const ColorKernel& k = kernels[index];
        const Vector3d&    u = colors.segment<3>(index * 3);
        grad(index) = k.calculate_squared_Mahalanobis_distance(u);
        grad.segment<3>(number_of_layers + index * 3) = 2.0 * alphas(index) * k.sigma_inv * (u - k.mu);
    }

    // Sparcity term
    double alpha_sum         = alphas.sum();
    double alpha_squared_sum = alphas.squaredNorm();
    for (int index = 0; index < number_of_layers; ++ index)
    {
        grad(index) += sigma * (alpha_squared_sum - 2.0 * alphas(index) * alpha_sum) / (alpha_squared_sum * alpha_squared_sum);
    }

    return grad;
}

double energy_function(const VectorXd& alphas, const VectorXd& colors, double sigma)
{
    const int number_of_layers = alphas.rows();

    // Main term
    double energy = 0.0;
    for (int index = 0; index < number_of_layers; ++ index)
    {
        energy += alphas[index] * kernels[index].calculate_squared_Mahalanobis_distance(colors.segment<3>(index * 3));
    }

    // Sparcity term
    energy += sigma * ((alphas.sum() / alphas.squaredNorm()) - 1.0);

    return energy;
}

double objective_function(const std::vector<double> &x, std::vector<double>& grad, void* data)
{
    const int number_of_layers = x.size() / 4;

    OptimizationParameterSet* set_pointer = static_cast<OptimizationParameterSet*>(data);

    const VectorXd alphas = Eigen::Map<const VectorXd>(&x[0], number_of_layers);
    const VectorXd colors = Eigen::Map<const VectorXd>(&x[number_of_layers], number_of_layers * 3);

    const Vector4d constraint_vector = calculate_equality_constraint_vector(alphas, colors, set_pointer->target_color);

    if (!grad.empty())
    {
        const VectorXd gradient_energy     = gradient_of_energy_function(alphas, colors, set_pointer->sigma);
        const VectorXd gradient_constraint = gradient_of_equality_constraint_terms(alphas, colors, set_pointer->target_color, constraint_vector, set_pointer->lambda, set_pointer->lo);
        Eigen::Map<VectorXd>(&grad[0], grad.size()) = gradient_energy + gradient_constraint;
    }

    return energy_function(alphas, colors, set_pointer->sigma) + calculate_equality_constraint_terms(constraint_vector, set_pointer->lambda, set_pointer->lo);
}

VectorXd solve_per_pixel_optimization(const Vector3d& target_color)
{
    const int number_of_layers = kernels.size();

    const VectorXd upper = VectorXd::Constant(number_of_layers * 4, 1.0);
    const VectorXd lower = VectorXd::Constant(number_of_layers * 4, 0.0);

    constexpr double gamma   = 0.25;
    constexpr double epsilon = 1e-08;
    constexpr double beta    = 10.0;

    OptimizationParameterSet set;
    set.lambda       = Vector4d::Constant(0.1);
    set.lo           = 0.1;
    set.target_color = target_color;
#ifdef SPARCITY
    set.sigma        = 10.0;
#else
    set.sigma        = 0.0;
#endif

#if 1
    VectorXd x_initial = VectorXd::Zero(number_of_layers * 4);

    double min_distance  = DBL_MAX;
    int    closest_index = - 1;
    for (int index = 0; index < number_of_layers; ++ index)
    {
        double distance = kernels[index].calculate_squared_Mahalanobis_distance(set.target_color);
        if (min_distance > closest_index)
        {
            min_distance  = distance;
            closest_index = index;
        }
    }
    x_initial(closest_index) = 1.0;
    for (int index = 0; index < number_of_layers; ++ index)
    {
        x_initial.segment<3>(number_of_layers + index * 3) = (index == closest_index) ? set.target_color : kernels[index].mu;
    }
#else
    VectorXd x_initial = VectorXd::Constant(number_of_layers * 4, 1.0 / static_cast<double>(number_of_layers));
    for (int index = 0; index < number_of_layers; ++ index)
    {
        x_initial.segment<3>(number_of_layers + index * 3) = target_color; // kernels[index].mu;
    }
#endif

    VectorXd x = x_initial;

    int count = 0;
    constexpr int max_count = 100;
    while (true)
    {
#ifdef GRADIENT
        const VectorXd x_new = nloptUtility::compute(x, upper, lower, objective_function, &set, nlopt::LD_MMA, 100);
#else
        const VectorXd x_new = nloptUtility::compute(x, upper, lower, objective_function, &set, nlopt::LN_COBYLA, 200);
#endif
        const Vector4d g     = calculate_equality_constraint_vector(x    , set.target_color);
        const VectorXd g_new = calculate_equality_constraint_vector(x_new, set.target_color);
        set.lambda += set.lo * g_new;

        if (g_new.norm() > gamma * g.norm()) set.lo *= beta;

        const bool is_changed = !x_new.isApprox(x, epsilon);
        x = x_new;

        if (!is_changed && g.norm() < epsilon) break;
        if (count > max_count)
        {
            std::cerr << "Error: g = [ " << g.transpose() << " ]" << std::endl;
            break;
        }

        ++ count;
    }
    return x;
}

QColor convert_to_qcolor(const Vector3d& color, double alpha)
{
    return QColor(color(0) * 255.0, color(1) * 255.0, color(2) * 255.0, alpha * 255.0);
}

}

void ColorUnmixing::compute_color_unmixing(const std::string &image_file_path, const std::string &output_directory_path)
{
    // Import the target image
    const QImage original_image(QString::fromStdString(image_file_path));
    const int width  = original_image.width();
    const int height = original_image.height();

    // Hardcode color models
    kernels.push_back(ColorKernel(Vector3d(1.0, 1.0, 1.0), Matrix3d::Identity()));
    kernels.push_back(ColorKernel(Vector3d(0.9, 0.8, 0.4), Matrix3d::Identity()));
    kernels.push_back(ColorKernel(Vector3d(0.5, 0.9, 0.2), Matrix3d::Identity()));
    kernels.push_back(ColorKernel(Vector3d(0.3, 0.1, 0.0), Matrix3d::Identity()));
    kernels.push_back(ColorKernel(Vector3d(0.0, 0.0, 0.0), Matrix3d::Identity()));

    const int number_of_layers = kernels.size();

    std::vector<QImage> layers(number_of_layers, QImage(QSize(width, height), QImage::Format_ARGB32));
    std::vector<QImage> overlay_layers(number_of_layers, QImage(QSize(width, height), QImage::Format_ARGB32));

    auto per_pixel_process = [&](int x, int y)
    {
        const QColor pixel_color = original_image.pixelColor(x, y);
        const VectorXd solution = solve_per_pixel_optimization(Vector3d(pixel_color.redF(), pixel_color.greenF(), pixel_color.blueF()));

        const VectorXd alphas = Eigen::Map<const VectorXd>(&solution[0], number_of_layers);
        const VectorXd colors = Eigen::Map<const VectorXd>(&solution[number_of_layers], number_of_layers * 3);

        for (int index = 0; index < number_of_layers; ++ index)
        {
            VectorXd overlay_alphas = VectorXd(number_of_layers);
            double sum_current_alpha = 0.0;
            for (int i = 0; i <= index; ++ i)
            {
                sum_current_alpha += alphas(i);
            }
            constexpr double epsilon = 1e-16;
            overlay_alphas(index) = (sum_current_alpha < epsilon) ? 0.0 : alphas(index) / sum_current_alpha;

            QColor color = convert_to_qcolor(colors.segment<3>(index * 3), alphas(index));
            layers[index].setPixelColor(x, y, color);

            QColor overlay_color = convert_to_qcolor(colors.segment<3>(index * 3), overlay_alphas(index));
            overlay_layers[index].setPixelColor(x, y, overlay_color);
        }
    };

#ifdef PARALLEL
    perform_in_parallel(per_pixel_process, width, height);
#else
    for (int x = 0; x < width; ++ x)
    {
        std::cout << x + 1 << " / " << width << std::endl;
        for (int y = 0; y < height; ++ y)
        {
            per_pixel_process(x, y);
        }
    }
#endif

    // Export layers
    for (int index = 0; index < number_of_layers; ++ index)
    {
        layers[index].save(QString::fromStdString(output_directory_path) + QString("/layer") + QString::number(index) + QString(".png"));
        overlay_layers[index].save(QString::fromStdString(output_directory_path) + QString("/overlay_layer") + QString::number(index) + QString(".png"));
    }
}

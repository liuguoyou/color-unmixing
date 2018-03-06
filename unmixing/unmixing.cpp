#include "unmixing.h"
#include <cmath>
#include <cfloat>
#include <iostream>
#include <thread>
#include <Eigen/LU>
#include "nloptutility.h"
#include "image_processing.h"

using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;
using Eigen::Vector3i;
using Eigen::Matrix3d;
using ImageProcessing::Image;
using ImageProcessing::ColorImage;

//#define SPARCITY
#define PARALLEL

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

struct ColorKernel
{
    ColorKernel(const Vector3d& mu, const Matrix3d& sigma_inv, int seed_x = - 1, int seed_y = - 1) :
        mu(mu),
        sigma_inv(sigma_inv),
        seed_x(seed_x),
        seed_y(seed_y)
    {
    }

    Vector3d mu;
    Matrix3d sigma_inv;
    int seed_x;
    int seed_y;

    double calculate_squared_Mahalanobis_distance(const Vector3d& color) const
    {
        return (color - mu).transpose() * sigma_inv * (color - mu);
    }
};

struct OptimizationParameterSet
{
    Vector3d target_color;
    Vector4d lambda;
    double   lo;
    double   sigma;
    std::vector<ColorKernel> kernels;
    bool     use_sparcity;
    bool     use_target_alphas; // If true, the alternative constraint (Eq. 6) will be used instead of the unity constraint (Eq. 2).
    VectorXd target_alphas;     // This will be used when "use_target_alphas" is true.
};

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

VectorXd gradient_of_equality_constraint_terms(const VectorXd& alphas,
                                               const VectorXd& colors,
                                               const Vector3d& target_color,
                                               const Vector4d& constraint_vector,
                                               const Vector4d& lambda,
                                               double lo,
                                               bool use_target_alphas,
                                               const VectorXd& target_alphas = VectorXd())
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
        const double   partial_g_a_per_partial_a = use_target_alphas ? 2.0 * (a - target_alphas(index)) : 2.0 * (sum_alpha - 1.0);

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

Vector4d calculate_equality_constraint_vector(const VectorXd& alphas,
                                              const VectorXd& colors,
                                              const Vector3d& target_color,
                                              bool use_target_alphas,
                                              const VectorXd& target_alphas = VectorXd())
{
    const Vector3d sum_color = composite_color(alphas, colors);
    const Vector3d g_color   = (sum_color - target_color).cwiseProduct(sum_color - target_color);
    const double   sum_alpha = alphas.sum();
    const double   g_alpha   = use_target_alphas ? (alphas - target_alphas).squaredNorm() : (sum_alpha - 1.0) * (sum_alpha - 1.0);

    return Vector4d(g_color(0), g_color(1), g_color(2), g_alpha);
}

Vector4d calculate_equality_constraint_vector(const VectorXd& x,
                                              const Vector3d& target_color,
                                              bool use_target_alphas,
                                              const VectorXd& target_alphas = VectorXd())
{
    const int number_of_layers = x.rows() / 4;
    return calculate_equality_constraint_vector(x.segment(0, number_of_layers), x.segment(number_of_layers, number_of_layers * 3), target_color, use_target_alphas, target_alphas);
}

// Calculate the gradient of the main objective function (Eq. 4)
VectorXd gradient_of_energy_function(const VectorXd& alphas,
                                     const VectorXd& colors,
                                     const std::vector<ColorKernel>& kernels,
                                     double sigma,
                                     bool use_sparcity)
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
    if (use_sparcity)
    {
        double alpha_sum         = alphas.sum();
        double alpha_squared_sum = alphas.squaredNorm();
        for (int index = 0; index < number_of_layers; ++ index)
        {
            grad(index) += sigma * (alpha_squared_sum - 2.0 * alphas(index) * alpha_sum) / (alpha_squared_sum * alpha_squared_sum);
        }
    }

    return grad;
}

// Calculate the main objective function (Eq. 4)
double energy_function(const VectorXd& alphas,
                       const VectorXd& colors,
                       const std::vector<ColorKernel>& kernels,
                       double sigma,
                       bool use_sparcity)
{
    const int number_of_layers = alphas.rows();

    // Main term
    double energy = 0.0;
    for (int index = 0; index < number_of_layers; ++ index)
    {
        energy += alphas[index] * kernels[index].calculate_squared_Mahalanobis_distance(colors.segment<3>(index * 3));
    }

    // Sparcity term
    if (use_sparcity) energy += sigma * ((alphas.sum() / alphas.squaredNorm()) - 1.0);

    return energy;
}

double objective_function(const std::vector<double> &x, std::vector<double>& grad, void* data)
{
    const int number_of_layers = x.size() / 4;

    const OptimizationParameterSet& set = *static_cast<const OptimizationParameterSet*>(data);

    const VectorXd alphas = Eigen::Map<const VectorXd>(&x[0], number_of_layers);
    const VectorXd colors = Eigen::Map<const VectorXd>(&x[number_of_layers], number_of_layers * 3);

    const Vector4d constraint_vector = calculate_equality_constraint_vector(alphas, colors, set.target_color, set.use_target_alphas, set.target_alphas);

    if (!grad.empty())
    {
        const VectorXd gradient_energy     = gradient_of_energy_function(alphas, colors, set.kernels, set.sigma, set.use_sparcity);
        const VectorXd gradient_constraint = gradient_of_equality_constraint_terms(alphas, colors, set.target_color, constraint_vector, set.lambda, set.lo, set.use_target_alphas, set.target_alphas);
        Eigen::Map<VectorXd>(&grad[0], grad.size()) = gradient_energy + gradient_constraint;
    }

    return energy_function(alphas, colors, set.kernels, set.sigma, set.use_sparcity) + calculate_equality_constraint_terms(constraint_vector, set.lambda, set.lo);
}

VectorXd solve_per_pixel_optimization(const Vector3d& target_color,
                                      const std::vector<ColorKernel>& kernels,
                                      bool for_refinement = false,
                                      const VectorXd& initial_colors = VectorXd(),
                                      const VectorXd& target_alphas = VectorXd())
{
    const int number_of_layers = kernels.size();

    const VectorXd upper = VectorXd::Constant(number_of_layers * 4, 1.0);
    const VectorXd lower = VectorXd::Constant(number_of_layers * 4, 0.0);

    constexpr double gamma   = 0.25;
    constexpr double epsilon = 1e-08;
    constexpr double beta    = 10.0;

    OptimizationParameterSet set;
    set.kernels           = kernels;
    set.lambda            = Vector4d::Constant(0.1);
    set.lo                = 0.1;
    set.target_color      = target_color;
    set.sigma             = 10.0;
    set.target_alphas     = target_alphas;
    if (!for_refinement)
    {
#ifdef SPARCITY
        set.use_sparcity      = true;
#else
        set.use_sparcity      = false;
#endif
        set.use_target_alphas = false;
    }
    else
    {
        set.use_sparcity      = false;
        set.use_target_alphas = true;
    }

    // Find an initial solution
    VectorXd x_initial = VectorXd::Zero(number_of_layers * 4);
    if (!for_refinement)
    {
        double min_distance  = DBL_MAX;
        int    closest_index = - 1;
        for (int index = 0; index < number_of_layers; ++ index)
        {
            double distance = kernels[index].calculate_squared_Mahalanobis_distance(set.target_color);
            if (min_distance > distance)
            {
                min_distance  = distance;
                closest_index = index;
            }
        }
        x_initial(closest_index) = 1.0;
        for (int index = 0; index < number_of_layers; ++ index)
        {
            x_initial.segment<3>(number_of_layers + index * 3) = (index == closest_index) ? set.target_color : kernels[index].mu;
            for (int i : { 0, 1, 2}) x_initial(number_of_layers + index * 3 + i) = std::max(std::min(x_initial(number_of_layers + index * 3 + i), 1.0), 0.0);
        }
    }
    else
    {
        x_initial.segment(0, number_of_layers) = target_alphas;
        x_initial.segment(number_of_layers, number_of_layers * 3) = initial_colors;
    }

    VectorXd x = x_initial;

    int count = 0;
    constexpr int max_count = 100;
    while (true)
    {
        const VectorXd x_new = nloptUtility::compute(x, upper, lower, objective_function, &set, nlopt::LD_MMA, 100, epsilon);
        const Vector4d g     = calculate_equality_constraint_vector(x    , set.target_color, for_refinement, target_alphas);
        const Vector4d g_new = calculate_equality_constraint_vector(x_new, set.target_color, for_refinement, target_alphas);

        set.lambda += set.lo * g_new;
        if (g_new.norm() > gamma * g.norm()) set.lo *= beta;
  
        const bool is_unchanged = (x_new - x).squaredNorm() < epsilon;
        const bool is_satisfied = g_new.norm() < epsilon;
        
        x = x_new;
        
        if ((is_unchanged && is_satisfied) || count > max_count) break;

        ++ count;
    }
    return x;
}

void print_kernel(const ColorKernel& kernel)
{
    std::cout << "mu: " << std::endl;
    std::cout << kernel.mu.transpose() << std::endl;
    std::cout << "sigma: " << std::endl;
    std::cout << kernel.sigma_inv.inverse() << std::endl;
    std::cout << "seed: " << std::endl;
    std::cout << "(" << kernel.seed_x << ", " << kernel.seed_y << ")" << std::endl;
}

void print_kernels(const std::vector<ColorKernel>& kernels)
{
    for (const ColorKernel& kernel : kernels)
    {
        std::cout << "---------------------" << std::endl;
        print_kernel(kernel);
    }
    std::cout << "---------------------" << std::endl;
}

void compute_normal_distribution(const ColorImage& original_image, const Image& weight_map, Vector3d& mu, Matrix3d& sigma)
{
    const int width  = original_image.width();
    const int height = original_image.height();

    mu = Vector3d::Zero();
    for (int x = 0; x < width; ++ x) for (int y = 0; y < height; ++ y)
    {
        const Vector3d I = original_image.get_rgb(x, y);
        mu += weight_map.get_pixel(x, y) * I;
    }
    sigma = Matrix3d::Zero();
    for (int x = 0; x < width; ++ x) for (int y = 0; y < height; ++ y)
    {
        const Vector3d I = original_image.get_rgb(x, y);
        sigma += weight_map.get_pixel(x, y) * (I - mu) * (I - mu).transpose();
    }

    // For avoiding singularity (note: this process is not used in the original paper)
    constexpr double epsilon = 1e-03; // This value is empirically set
    sigma += epsilon * Matrix3d::Identity();
}

std::vector<ColorImage> perform_matte_refinement(const ColorImage& original_image, const std::vector<ColorImage>& layers, const std::vector<ColorKernel>& kernels)
{
    assert(layers.size() == kernels.size());

    const int number = layers.size();
    const int width  = original_image.width();
    const int height = original_image.height();
    const int radius = 60 * std::min(width, height) / 1000;
    constexpr double epsilon = 1e-04;

    // Apply guided filter
    std::vector<Image> refined_alphas;
    for (const ColorImage& layer : layers)
    {
        const Image alpha = layer.get_a();
        const Image refined_alpha = ImageProcessing::apply_guided_filter(alpha, original_image, radius, epsilon);

        refined_alphas.push_back(refined_alpha);
    }

    // Regularize alphas such that the sum equals to one for each pixel
    for (int x = 0; x < width; ++ x) for (int y = 0; y < height; ++ y)
    {
        double sum = 0.0;
        for (int i = 0; i < number; ++ i)
        {
            refined_alphas[i].set_pixel(x, y, std::max(std::min(refined_alphas[i].get_pixel(x, y), 1.0), 0.0));
            sum += refined_alphas[i].get_pixel(x, y);
        }
        assert(sum > 0.0);
        for (int i = 0; i < number; ++ i)
        {
            refined_alphas[i].set_pixel(x, y, refined_alphas[i].get_pixel(x, y) / sum);
        }
    }

    // Perform optimization
    std::vector<ColorImage> refined_layers(number, ColorImage(width, height));
    auto per_pixel_process = [&](int x, int y)
    {
        VectorXd initial_colors(number * 3);
        VectorXd target_alphas(number);
        for (int i = 0; i < number; ++ i)
        {
            initial_colors.segment<3>(i * 3) = layers[i].get_rgb(x, y);
            target_alphas(i) = refined_alphas[i].get_pixel(x, y);
        }

        const Vector3d pixel_color = original_image.get_rgb(x, y);
        const VectorXd solution = solve_per_pixel_optimization(pixel_color, kernels, true, initial_colors, target_alphas);

        const VectorXd alphas = Eigen::Map<const VectorXd>(&solution[0], number);
        const VectorXd colors = Eigen::Map<const VectorXd>(&solution[number], number * 3);

        for (int index = 0; index < number; ++ index)
        {
            refined_layers[index].set_rgba(x, y, colors.segment<3>(index * 3), alphas(index));
        }
    };
#ifdef PARALLEL
    perform_in_parallel(per_pixel_process, width, height);
#else
    for (int x = 0; x < width; ++ x) for (int y = 0; y < height; ++ y) per_pixel_process(x, y);
#endif

    return refined_layers;
}

std::vector<ColorImage> convert_alpha_add_to_overlay(const std::vector<ColorImage>& layers)
{
    const int number = layers.size();
    const int width  = layers.front().width();
    const int height = layers.front().height();

    std::vector<ColorImage> overlay_layers(number, ColorImage(width, height));

    for (int x = 0; x < width; ++ x) for (int y = 0; y < height; ++ y)
    {
        for (int index = 0; index < number; ++ index)
        {
            VectorXd overlay_alphas = VectorXd(number);
            double sum_current_alpha = 0.0;
            for (int i = 0; i <= index; ++ i)
            {
                sum_current_alpha += layers[i].get_a().get_pixel(x, y);
            }
            constexpr double epsilon = 1e-16;
            overlay_alphas(index) = (sum_current_alpha < epsilon) ? 1.0 : layers[index].get_a().get_pixel(x, y) / sum_current_alpha;
            overlay_layers[index].set_rgba(x, y, layers[index].get_rgb(x, y), overlay_alphas(index));
        }
    }
    return overlay_layers;
}

}

void ColorUnmixing::compute_color_unmixing(const std::string &image_file_path, const std::string &output_directory_path)
{
    // Import the target image
    const ColorImage original_image(image_file_path);
    const int width  = original_image.width();
    const int height = original_image.height();

    // Calculate intermediate images
    const Image gray_image         = original_image.get_luminance();
    const Image gradient_magnitude = ImageProcessing::calculate_gradient_magnitude(gray_image);

    // Compute color models
    std::vector<ColorKernel> kernels;
    std::vector<std::vector<bool>> well_represented(width, std::vector<bool>(height, false));

    constexpr double tau                 = 5.0; // The value used in the original paper is 5.
    constexpr int    neighborhood_radius = 10;  // In the paper, this value is fixed to 10 (i.e., 20 x 20 neighborhood) for any input image. This may need to be modified so that it adapts to the image size.
    constexpr int    number_of_bins      = 10;
    
    while (true)
    {
        // Initialize bins
        double bins[number_of_bins][number_of_bins][number_of_bins];
        std::fill(bins[0][0], bins[number_of_bins][0], 0.0);

        auto get_bin = [number_of_bins, &original_image](int x, int y)
        {
            const Vector3d pixel_color = original_image.get_rgb(x, y);
            const int bin_r = std::min(static_cast<int>(std::floor(pixel_color(0) * number_of_bins)), number_of_bins - 1);
            const int bin_g = std::min(static_cast<int>(std::floor(pixel_color(1) * number_of_bins)), number_of_bins - 1);
            const int bin_b = std::min(static_cast<int>(std::floor(pixel_color(2) * number_of_bins)), number_of_bins - 1);
            return Vector3i(bin_r, bin_g, bin_b);
        };

        // Poll bins
        auto per_pixel_polling_process = [&](int x, int y)
        {
            // Skip if the pixel is already well represented
            if (well_represented[x][y]) return;

            // Calculate bin
            const Vector3i bin = get_bin(x, y);

            // Calculate per-pixel representation score
            double representation_score = DBL_MAX;
            if (!kernels.empty())
            {
                const Vector3d pixel_color = original_image.get_rgb(x, y);
                const VectorXd solution = solve_per_pixel_optimization(pixel_color, kernels);
                const VectorXd alphas = Eigen::Map<const VectorXd>(&solution[0], kernels.size());
                const VectorXd colors = Eigen::Map<const VectorXd>(&solution[kernels.size()], kernels.size() * 3);

                representation_score = energy_function(alphas, colors, kernels, 0.0, false);
            }

            // Reject if it is already well represented
            if (representation_score < tau * tau)
            {
                well_represented[x][y] = true;
                return;
            }

            // Calculate vote values
            const double vote_weight = std::exp(- gradient_magnitude.get_pixel(x, y)) * (1.0 - std::exp(- representation_score));

            // Vote to bin
            bins[bin(0)][bin(1)][bin(2)] += vote_weight;
        };
#ifdef PARALLEL
        perform_in_parallel(per_pixel_polling_process, width, height);
#else
        for (int x = 0; x < width; ++ x) for (int y = 0; y < height; ++ y) per_pixel_polling_process(x, y);
#endif
        
        // Export the current mask
        Image well_represented_map(width, height, 0.0);
        for (int x = 0; x < width; ++ x) for (int y = 0; y < height; ++ y)
        {
            well_represented_map.set_pixel(x, y, (well_represented[x][y] ? 1.0 : 0.0));
        }
        well_represented_map.save(output_directory_path + "/rep" + std::to_string(kernels.size()) + ".png");

        // Break the loop if *almost* all the pixels are already well represented
        int count = 0;
        for (int x = 0; x < width; ++ x) for (int y = 0; y < height; ++ y)
        {
            if (well_represented[x][y]) ++ count;
        }
        std::cout << count << " / " << width * height << std::endl;
        constexpr double almost_threshold = 0.995;
        const bool done = (almost_threshold < static_cast<double>(count) / static_cast<double>(width * height));
        if (done) break;

        // Select the most popular bin
        Vector3i max_bin = Vector3i::Constant(- 1);
        double max_bin_vote = 0.0;
        for (int r = 0; r < number_of_bins; ++ r)
        {
            for (int g = 0; g < number_of_bins; ++ g)
            {
                for (int b = 0; b < number_of_bins; ++ b)
                {
                    if (max_bin_vote <= bins[r][g][b])
                    {
                        max_bin = Vector3i(r, g, b);
                        max_bin_vote = bins[r][g][b];
                    }
                }
            }
        }

        // Select seed pixel
        int seed_x = - 1;
        int seed_y = - 1;
        double max_score = 0.0;
        for (int x = 0; x < width; ++ x) for (int y = 0; y < height; ++ y)
        {
            // Ignore if it is well represented already
            if (well_represented[x][y]) continue;

            // Calculate bin
            const Vector3i bin = get_bin(x, y);

            // Ignore if the pixel does not belong to the selected bin
            if (bin != max_bin) continue;

            // Calculate score
            int neighborhood_count = 0;
            for (int offset_x = - neighborhood_radius; offset_x <= neighborhood_radius; ++ offset_x)
            {
                for (int offset_y = - neighborhood_radius; offset_y <= neighborhood_radius; ++ offset_y)
                {
                    if (x + offset_x < 0 || x + offset_x >= width)  continue;
                    if (y + offset_y < 0 || y + offset_y >= height) continue;
                    if (well_represented[x + offset_x][y + offset_y]) continue;

                    const Vector3i neighbor_bin = get_bin(x + offset_x, y + offset_y);

                    if (neighbor_bin == max_bin) ++ neighborhood_count;
                }
            }
            assert(neighborhood_count > 0);
            const double score = static_cast<double>(neighborhood_count) * std::exp(- gradient_magnitude.get_pixel(x, y));

            // Update the seed pixel candidate
            if (max_score < score)
            {
                max_score = score;
                seed_x = x;
                seed_y = y;
            }
        }
        assert(seed_x >= 0 && seed_y >= 0);

        // Compute guided filter weights
        // Note: preventing the values from being negative is necessary to ensure the validity of the obtained normal distribution
        const Image weight_map = ImageProcessing::calculate_guided_filter_kernel(gray_image, seed_x, seed_y, neighborhood_radius);

        // Export the weight map
        Image temporary_weight_map = weight_map;
        temporary_weight_map.scale_to_unit();
        temporary_weight_map.save(output_directory_path + "/weight" + std::to_string(kernels.size()) + ".png");

        // Calculate the color distribution
        Vector3d mu;
        Matrix3d sigma;
        compute_normal_distribution(original_image, weight_map, mu, sigma);

        // Add a new color kernel
        const ColorKernel kernel = ColorKernel(mu, sigma.inverse(), seed_x, seed_y);
        print_kernel(kernel);
        kernels.push_back(kernel);
    }

    const int number_of_layers = kernels.size();

    std::vector<ColorImage> layers(number_of_layers, ColorImage(width, height));

    auto per_pixel_process = [&](int x, int y)
    {
        const Vector3d pixel_color = original_image.get_rgb(x, y);
        const VectorXd solution = solve_per_pixel_optimization(pixel_color, kernels);

        const VectorXd alphas = Eigen::Map<const VectorXd>(&solution[0], number_of_layers);
        const VectorXd colors = Eigen::Map<const VectorXd>(&solution[number_of_layers], number_of_layers * 3);

        for (int index = 0; index < number_of_layers; ++ index)
        {
            layers[index].set_rgba(x, y, colors.segment<3>(index * 3), alphas(index));
        }
    };

#ifdef PARALLEL
    perform_in_parallel(per_pixel_process, width, height);
#else
    for (int x = 0; x < width; ++ x) for (int y = 0; y < height; ++ y) per_pixel_process(x, y);
#endif

    const std::vector<ColorImage> overlay_layers = convert_alpha_add_to_overlay(layers);
    const std::vector<ColorImage> refined_layers = perform_matte_refinement(original_image, layers, kernels);
    const std::vector<ColorImage> refined_overlay_layers = convert_alpha_add_to_overlay(refined_layers);

    // Export layers
    for (int index = 0; index < number_of_layers; ++ index)
    {
        layers[index].save(output_directory_path + "/layer" + std::to_string(index) + ".png");
        layers[index].get_a().save(output_directory_path + "/layer_alpha" + std::to_string(index) + ".png");
        overlay_layers[index].save(output_directory_path + "/overlay_layer" + std::to_string(index) + ".png");
        refined_layers[index].save(output_directory_path + "/refined_layer" + std::to_string(index) + ".png");
        refined_layers[index].get_a().save(output_directory_path + "/refined_layer_alpha" + std::to_string(index) + ".png");
        refined_overlay_layers[index].save(output_directory_path + "/refined_overlay_layer" + std::to_string(index) + ".png");
    }

    // Export the original image
    original_image.save(output_directory_path + "/original.png");

    // Print kernel info
    print_kernels(kernels);
}
